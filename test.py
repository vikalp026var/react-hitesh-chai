def test_indicator_value_invalid_cases(simulation):
    """Test that indicator_value method returns None for invalid cases."""
    assert simulation.data.indicator_value('^SPX', 'uahfsufdsfa') is None
    assert (
        simulation.data.indicator_value('^SPX', 'bbands(15,2)', 'qwerty')
        is None
    )
    assert simulation.data.indicator_value('^SPX', 'bbands(15,2)') is None


SUPPORTED_INDICATORS = [
    item for ele in ta.Category.values() for item in ele
] + list_indicators()





# data 
# Standard library
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import starlink.supplementary

from . import technical_indicators, utils
from .aws import rds
from .constants import constants

# Local
from .default import default
from .objects import Option
from . import slindicators

# Optional dependencies
try:
    import ib_api

    IB_API_AVAILABLE = True
except ModuleNotFoundError:
    IB_API_AVAILABLE = False

try:
    from . import stochastic

    STOCHASTIC_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    STOCHASTIC_AVAILABLE = False


class DataAccess:
    """
    Simplified data access for v3 simulations.

    Wraps HybridClient and provides high-level methods for fetching
    stock prices, indicators, and option contracts. Used by Simulation,
    BatchSimulation, and Optimization classes.

    Parameters
    ----------
    simulation : Simulation
        Simulation instance providing configuration and state

    Examples
    --------
    >>> data = DataAccess(simulation)
    >>> price = data.stock_price('SPY', '2023-01-15 14:30:00')
    >>> contracts = data.find_contracts(contract_info, dte, delta=delta_range)
    """

    def __init__(self, simulation, client=None):
        """
        Initialize data access with HybridClient.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance providing configuration and state
        client : HybridClient, optional
            Existing HybridClient instance to reuse. If None, creates a new one.
        """
        self.context = simulation

        # Use provided client or create new HybridClient
        if client is not None:
            self.client = client
        else:
            self.client = rds.get_hybrid_client(
                client_type=simulation.client_type
            )

    def stock_price(self, symbol: str, datetime: Optional[str] = None) -> float:
        """
        Get stock price for a symbol at a specific datetime.

        Parameters
        ----------
        symbol : str
            Stock symbol (e.g., 'SPY')
        datetime : str, optional
            Datetime string in format 'YYYY-MM-DD HH:MM:SS'.
            If None, uses current simulation datetime.

        Returns
        -------
        float
            Stock price

        Raises
        ------
        ValueError
            If datetime is in the future

        Examples
        --------
        >>> price = self.sim.stock_price('SPY')
        >>> price = self.sim.stock_price('SPY', '2023-01-15 14:30:00')
        """
        ctx = self.context
        dt = datetime or ctx.cur_datetime

        if dt > ctx.cur_datetime:
            raise ValueError(
                f'Provided datetime {dt} is greater than the current datetime {ctx.cur_datetime}.'
            )

        # Check cached price data first
        price = ctx.price_data.get(symbol, {}).get(dt, None)

        if price is None and ctx.derivative_type == 'equity_index_etf':
            # Fetch from HybridClient
            quotes = self.client.get_stock_quotes(
                symbol,
                dt,
                dt,
                ctx.daily_timestamp if ctx.sim_interval == 'daily' else None,
                columns=['mid'],
            )
            if quotes:
                price = quotes.get(dt, {}).get('mid', None)

        elif price is None and ctx.derivative_type == 'futures':
            # Fetch futures quotes
            quotes = self.client.get_futures_quotes(
                symbol,
                dt,
                dt,
                ctx.daily_timestamp if ctx.sim_interval == 'daily' else None,
                columns=[default.FUTURES_PRICE_TYPE],
            )
            if quotes:
                price = quotes.get(dt, {}).get(default.FUTURES_PRICE_TYPE)

        if price is None:
            print(
                f'Warning: price is not available for symbol {symbol} on {dt}'
            )

        return price

    def indicator_value(
        self,
        symbol: str,
        indicator: str,
        attribute: Optional[str] = None,
        datetime: Optional[str] = None,
    ) -> float:
        """
        Get technical indicator value.

        Parameters
        ----------
        symbol : str
            Stock symbol
        indicator : str
            Indicator name (e.g., 'sma(20)', 'rsi(14)')
        attribute : str, optional
            Specific attribute for complex indicators
            (e.g., 'upper' for bbands)
        datetime : str, optional
            Datetime string. If None, uses current simulation datetime.

        Returns
        -------
        float or None
            Indicator value at datetime, or None if not found

        Examples
        --------
        >>> sma = self.sim.indicator_value('SPY', 'sma(20)')
        >>> rsi = self.sim.indicator_value('SPY', 'rsi(14)')
        >>> bb_upper = self.sim.indicator_value('SPY', 'bbands(20,2)', 'upper')
        """
        ctx = self.context
        dt = datetime or ctx.cur_datetime

        # Get indicators for this symbol/datetime
        indicators_on_date = ctx.feature_data.get(symbol, {}).get(dt, {})
        if not indicators_on_date:
            return None

        # Normalize indicator name (remove spaces)
        indicator = indicator.replace(' ', '')

        # Find matching indicators
        matching_indicators = [
            e for e in indicators_on_date.keys() if indicator in e
        ]

        if not matching_indicators:
            if ctx.debug:
                print(
                    f'Indicator {indicator} not found on {dt} for {symbol}. '
                    f'Valid indicators are {list(indicators_on_date.keys())}'
                )
            return None

        # Build indicator key with attribute if provided
        indicator_key = f'{indicator}_{attribute}' if attribute else indicator

        if indicator_key in matching_indicators:
            return indicators_on_date.get(indicator_key, None)
        else:
            if ctx.debug:
                print(
                    f"Attribute '{attribute}' not found for indicator {indicator}. "
                    f'Valid attributes are {[e.split("_")[1] for e in matching_indicators]}'
                )
            return None

    def find_contract(
        self,
        symbol: str,
        option_type: str,
        quantity: int,
        filter_criteria: dict,
        estimated: bool = False,
    ) -> pd.DataFrame:
        """
        Search for option contracts matching specified filtering criteria.

        .. warning::
            **DEPRECATED**: Use ``find_contracts`` instead for new code.

            This method is maintained for backward compatibility with legacy strategies.

        Parameters
        ----------
        symbol : str
            Underlying asset symbol (e.g., 'SPY', '^SPX')
        option_type : str
            'C'/'call' or 'P'/'put'
        quantity : int
            Number of contracts (used for position sizing)
        filter_criteria : dict
            DynamoDB-style filter structure with keys:
            - 'ConditionalOperator': 'AND' or 'OR'
            - 'Conditions': dict of filter conditions
              - 'days_to_expiration': {'ComparisonOperator': 'BETWEEN', 'lower': int, 'upper': int}
              - 'delta': {'ComparisonOperator': 'BETWEEN', 'lower': float, 'upper': float}
              - 'strike': {'ComparisonOperator': 'BETWEEN', 'lower': float, 'upper': float}
        estimated : bool, optional
            If True, returns estimated contracts (for stochastic simulations)

        Returns
        -------
        pd.DataFrame
            DataFrame with matching contracts containing columns:
            - contract_id, strike, expiration, option_type
            - days_to_expiration, delta, gamma, vega, theta
            - implied_volatility, bid, ask, mid, underlying_mid
            Returns empty DataFrame if no matches found.

        Examples
        --------
        >>> filter_criteria = {
        ...     'ConditionalOperator': 'AND',
        ...     'Conditions': {
        ...         'days_to_expiration': {'ComparisonOperator': 'BETWEEN', 'lower': 20, 'upper': 30},
        ...         'delta': {'ComparisonOperator': 'BETWEEN', 'lower': -0.30, 'upper': -0.20}
        ...     }
        ... }
        >>> contracts = data.find_contract('SPY', 'P', 10, filter_criteria)
        """
        ctx = self.context

        # Handle estimated/stochastic contracts
        if estimated or ctx.is_stochastic:
            return self._find_estimated_contracts(
                symbol, option_type, quantity, filter_criteria
            )

        # Historical backtesting
        if not ctx.is_live_trading:
            return self.client.legacy_option_query(
                underlying_symbol=symbol,
                quote_datetime=ctx.cur_datetime,
                option_type=option_type,
                query_args=filter_criteria,
                columns=None,
                return_pandas=True,
                data_type=ctx.derivative_type,
            )

        # Live trading via IB API
        else:
            if not IB_API_AVAILABLE:
                raise RuntimeError(
                    'ib_api module not found. Install it for live trading support.'
                )

            api = ib_api.IbApi(ctx.live_trading_fund)
            polygon_filter = utils.convert_ddb_filter_to_polygon_filter(
                symbol, option_type, quantity, filter_criteria
            )
            contracts, *_ = api.find_contracts(
                {'position': polygon_filter},
                [None],
                use_ibkr=False,
                default_to_min_exp=False,
            )

            # Reformat to match historical data schema
            if not contracts['position'].empty:
                df = (
                    contracts['position']
                    .reset_index()
                    .rename(
                        columns={
                            'dte': 'days_to_expiration',
                            'put_call': 'option_type',
                            'expiration_date': 'expiration',
                        }
                    )
                    .drop(columns=['name'])
                )
                # Add underlying_mid for compatibility
                df['underlying_mid'] = df['underlying']
                return df

            return contracts['position']

    def _find_estimated_contracts(
        self,
        symbol: str,
        option_type: str,
        quantity: int,
        filter_criteria: dict,
    ) -> pd.DataFrame:
        """
        Find estimated option contracts (for stochastic simulations).

        Used internally when running stochastic simulations or when estimated=True
        is passed to find_contract.

        Parameters
        ----------
        symbol : str
            Underlying symbol
        option_type : str
            'C' or 'P'
        quantity : int
            Number of contracts
        filter_criteria : dict
            Filter criteria with DTE and delta/strike targets

        Returns
        -------
        pd.DataFrame
            DataFrame with estimated contract details
        """
        conditions = filter_criteria.get('Conditions', {})

        # Validate required fields
        if 'days_to_expiration' not in conditions:
            raise ValueError('days_to_expiration and target must be provided')

        has_delta = 'delta' in conditions
        has_strike = 'strike' in conditions

        if not (has_delta != has_strike):  # XOR check
            raise ValueError(
                'Either delta OR strike must be provided (not both)'
            )

        if has_delta and conditions['delta'].get('target') is None:
            raise ValueError(
                'Delta target must be provided when using delta filter'
            )

        if has_strike and conditions['strike'].get('target') is None:
            raise ValueError(
                'Strike target must be provided when using strike filter'
            )

        ctx = self.context

        # Build position for Option object
        position = {
            'option_type': option_type,
            'name': '_',
            'quantity': quantity,
            'type': 'option',
            'underlying_symbol': symbol,
            'root': symbol,
            'days_to_expiration': int(
                conditions['days_to_expiration']['target']
            ),
            'tracking_price_type': 'estimated',
            'contract_id': f'{symbol}/{utils.future_date(ctx.cur_datetime[:10], int(conditions["days_to_expiration"]["target"]))}/{option_type}/9999',
        }

        # Get position prices (underlying data for estimated options)
        position['quotes'] = self.get_position_prices(position)

        if has_delta:
            position['init_delta'] = conditions['delta']['target']
        elif has_strike:
            position['init_strike'] = conditions['strike']['target']

        # Create Option object to calculate prices
        option = Option(position, ctx.cur_datetime, ctx.runtime_config)
        option.open_position(ctx.cur_datetime)

        # Return DataFrame matching RDS schema
        return pd.DataFrame(
            {
                'underlying_symbol': option.underlying_symbol,
                'quote_datetime': ctx.cur_datetime,
                'root': option.root,
                'expiration': option.expiration,
                'strike': option.strike,
                'option_type': option.option_type,
                'open': option.stats[ctx.cur_datetime]['price'],
                'high': option.stats[ctx.cur_datetime]['price'],
                'low': option.stats[ctx.cur_datetime]['price'],
                'close': option.stats[ctx.cur_datetime]['price'],
                'trade_volume': 1,
                'bid_size': 1,
                'bid': option.stats[ctx.cur_datetime]['price'],
                'ask_size': 1,
                'ask': option.stats[ctx.cur_datetime]['price'],
                'implied_volatility': option.stats[ctx.cur_datetime]['iv'],
                'delta': option.stats[ctx.cur_datetime]['delta'],
                'gamma': option.stats[ctx.cur_datetime]['gamma'],
                'theta': option.stats[ctx.cur_datetime]['theta'],
                'vega': option.stats[ctx.cur_datetime]['vega'],
                'rho': option.stats[ctx.cur_datetime]['rho'],
                'open_interest': 1,
                'quote_date': ctx.cur_datetime.split(' ')[0],
                'underlying_mid': option.stats[ctx.cur_datetime][
                    'underlying_price'
                ],
                'mid': option.stats[ctx.cur_datetime]['price'],
                'days_to_expiration': option.days_to_expiration,
                'contract_id': option.contract_id,
            },
            index=[0],
        )

    def find_contracts(
        self,
        contract_info: dict,
        days_to_expiration: dict,
        delta: Optional[dict] = None,
        strike: Optional[dict] = None,
        return_best_contract: bool = False,
    ):
        """
        Find option contracts matching criteria.

        Simplified version adapted from data_access.py for v3.
        Delegates actual search to data source.

        Parameters
        ----------
        contract_info : dict
            Contract specification with keys:
            - name: Position name
            - type: 'option', 'futures_option', 'stock', 'futures_contract'
            - underlying_symbol: Underlying symbol
            - quantity: Number of contracts
            - option_type: 'C' or 'P' (for options)
        days_to_expiration : dict
            DTE range: {'min': int, 'max': int, 'target': int}
        delta : dict, optional
            Delta range: {'min': float, 'max': float, 'target': float}
        strike : dict, optional
            Strike range: {'min': float, 'max': float, 'target': float}
        return_best_contract : bool
            If True, return single best match; if False, return all matches

        Returns
        -------
        dict or pd.DataFrame or None
            - dict: If return_best_contract=True
            - DataFrame: If return_best_contract=False
            - None: If no contracts found or validation fails
        """

        # Validate required fields
        if not all(
            k in contract_info
            for k in ['name', 'type', 'underlying_symbol', 'quantity']
        ):
            print(
                'Missing required fields (name, type, underlying_symbol, quantity) in contract_info'
            )
            return None

        leg_type = contract_info['type']
        underlying = contract_info['underlying_symbol']
        quantity = contract_info['quantity']

        # Validate leg type
        if leg_type not in [
            'option',
            'futures_option',
            'stock',
            'futures_contract',
        ]:
            print(f'Invalid leg type provided: {leg_type}')
            return None

        # Handle stock contracts
        if leg_type == 'stock':
            return self._find_stock_contract(contract_info)

        # Options require delta or strike
        if (
            leg_type in ['option', 'futures_option']
            and not delta
            and not strike
        ):
            print(
                'No delta or strike configuration provided for option contract'
            )
            return None

        # Validate DTE configuration
        if not self._validate_selection_config(
            'days_to_expiration', days_to_expiration
        ):
            return None

        # Validate delta/strike configuration
        if leg_type in ['option', 'futures_option']:
            selection_key_name = 'delta' if delta else 'strike'
            selection_key_config = delta if delta else strike
            if not self._validate_selection_config(
                selection_key_name, selection_key_config
            ):
                return None

        # Build filter criteria
        filter_criteria = self._build_filter_criteria(
            days_to_expiration, delta, strike
        )

        # Handle futures contracts
        if leg_type == 'futures_contract':
            df = self._find_futures_contracts(underlying, filter_criteria)

        # Handle futures options
        elif leg_type == 'futures_option':
            df = self._find_futures_options(
                underlying,
                contract_info.get('option_type'),
                days_to_expiration,
                filter_criteria,
            )

        # Handle equity options
        else:
            df = self._find_equity_options(
                underlying,
                contract_info.get('option_type'),
                quantity,
                filter_criteria,
            )

        if df is None or df.empty:
            return None

        # Return all contracts if not selecting best
        if not return_best_contract:
            return df

        # Select best contract
        dte_target = days_to_expiration.get('target')
        delta_target = delta.get('target') if delta else None
        strike_target = strike.get('target') if strike else None

        return self._select_best_contract(
            df, contract_info, dte_target, delta_target, strike_target
        )

    def _find_stock_contract(self, contract_info: dict) -> Optional[dict]:
        """Find and format stock contract."""
        ctx = self.context
        underlying = contract_info['underlying_symbol']

        query = f"""
        select *
        from quotes.hourly_underlying
        where quote_datetime='{ctx.cur_datetime}'
        and underlying_symbol='{underlying}'
        and mid > 0
        """

        df = self.client.query(query)
        if df.empty:
            return None

        contract = df.iloc[0].to_dict()
        return {
            **contract,
            'name': contract_info['name'],
            'type': contract_info['type'],
            'quantity': contract_info['quantity'],
            'qty': contract_info['quantity'],
            'underlying_symbol': underlying,
            'info': contract_info.get('info', {}),
            'profit': 0,
            'days_in_trade': 0,
            'opening_price': contract['mid'],
        }

    def _validate_selection_config(self, name: str, config: dict) -> bool:
        """
        Validate selection configuration (DTE, delta, or strike).

        Note: 'target' is optional - it's only used for selecting the best contract
        from multiple matches. 'min' and 'max' are required for range filtering.
        """
        if not config:
            print(f'No {name} configuration provided')
            return False

        # Only validate min/max - target is optional
        for key in ['min', 'max']:
            value = config.get(key)
            if value is None:
                print(f'Invalid {name} range provided - missing {key}')
                return False

        return True

    def _build_filter_criteria(
        self, dte: dict, delta: Optional[dict], strike: Optional[dict]
    ) -> dict:
        """Build filter criteria for option search."""
        criteria = {
            'ConditionalOperator': 'AND',
            'Conditions': {
                'days_to_expiration': {
                    'ComparisonOperator': 'BETWEEN',
                    'lower': dte['min'],
                    'upper': dte['max'],
                }
            },
        }

        if delta:
            criteria['Conditions']['delta'] = {
                'ComparisonOperator': 'BETWEEN',
                'lower': delta['min'],
                'upper': delta['max'],
            }
        elif strike:
            criteria['Conditions']['strike'] = {
                'ComparisonOperator': 'BETWEEN',
                'lower': strike['min'],
                'upper': strike['max'],
            }

        return criteria

    def _find_equity_options(
        self,
        underlying: str,
        option_type: str,
        quantity: int,
        filter_criteria: dict,
    ) -> pd.DataFrame:
        """Find equity option contracts."""
        # Delegate to find_contract for consistency
        return self.find_contract(
            underlying, option_type, quantity, filter_criteria
        )

    def _find_futures_contracts(
        self, underlying: str, filter_criteria: dict
    ) -> pd.DataFrame:
        """Find futures contracts."""
        ctx = self.context

        return self.client.legacy_option_query(
            underlying_symbol=underlying,
            quote_datetime=ctx.cur_datetime,
            option_type=None,
            query_args=filter_criteria,
            columns=None,
            return_pandas=True,
            data_type='futures',
        )

    def _find_futures_options(
        self,
        underlying: str,
        option_type: str,
        dte: dict,
        filter_criteria: dict,
    ) -> Optional[pd.DataFrame]:
        """Find futures option contracts."""
        ctx = self.context

        futures_symbol = self.get_futures_symbol(
            underlying, dte, ctx.cur_datetime
        )
        if not futures_symbol:
            return None

        futures_symbol = (
            futures_symbol
            if type(futures_symbol) is not list
            else futures_symbol[0]
        )

        # Add futures symbol filter
        filter_criteria['Conditions']['futures_symbol'] = {
            'ComparisonOperator': 'EQ',
            'value': futures_symbol,
        }

        return self.client.legacy_option_query(
            underlying_symbol=underlying,
            quote_datetime=ctx.cur_datetime,
            option_type=option_type,
            query_args=filter_criteria,
            columns=None,
            return_pandas=True,
            data_type='futures_options',
        )

    def _select_best_contract(
        self,
        df: pd.DataFrame,
        contract_info: dict,
        dte_target: int,
        delta_target: Optional[float] = None,
        strike_target: Optional[float] = None,
    ) -> Optional[dict]:
        """Select best contract from DataFrame based on target criteria.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of pre-filtered contract candidates
        contract_info : dict
            Contract specification with type, quantity, etc.
        dte_target : int
            Target days to expiration
        delta_target : float, optional
            Target delta value (for options)
        strike_target : float, optional
            Target strike price (for options)

        Returns
        -------
        dict or None
            Best matching contract or None if no valid match

        Notes
        -----
        Assumes df is already filtered by min/max ranges.
        Only one of delta_target or strike_target should be provided for options.
        """
        # First filter to best expiration
        best_exp = df.iloc[
            int((df['days_to_expiration'] - dte_target).abs().idxmin())
        ]['expiration']
        df = df.loc[df['expiration'] == best_exp].reset_index(drop=True)

        # Select based on delta or strike target
        leg_type = contract_info['type']
        if leg_type in ['option', 'futures_option']:
            if delta_target is not None:
                selection_key = 'delta'
                target = delta_target
            elif strike_target is not None:
                selection_key = 'strike'
                target = strike_target
            else:
                raise ValueError(
                    f'Either delta target or strike target must be provided for {leg_type} contracts when automatically selecting the best contract'
                )

            contract_series = df.iloc[
                int((df[selection_key] - target).abs().idxmin())
            ]
        else:
            contract_series = df.iloc[0]

        # Get opening price
        opening_price = contract_series.get(
            'mid',
            contract_series.get(
                'close',
                contract_series.get(
                    'settlement_price', contract_series.get('open_price')
                ),
            ),
        )

        # Build best contract dict
        best_contract = {
            **contract_series.to_dict(),
            'name': contract_info['name'],
            'type': leg_type,
            'quantity': contract_info['quantity'],
            'qty': contract_info['quantity'],
            'underlying_symbol': contract_info['underlying_symbol'],
            'info': contract_info.get('info', {}),
            'profit': 0,
            'days_in_trade': 0,
            'opening_price': opening_price,
        }

        # Validate mid price
        return best_contract if best_contract.get('mid', 0) > 0 else None

    # ==================== Position Data Methods ====================

    def get_position_prices(self, position: dict) -> Optional[dict]:
        """
        Get price quotes for a position based on its type.

        Retrieves current market prices for the specified position. For options with
        estimated pricing, also updates volatility surfaces. Handles multiple position
        types including stocks, options, futures contracts, and futures options.

        Parameters
        ----------
        position : dict
            Position dictionary containing:
            - type : str - 'stock', 'option', 'futures_contract', 'futures_option'
            - tracking_price_type : str, optional - For options, 'estimated' uses surfaces
            - underlying_symbol : str - Symbol of underlying asset
            - Additional type-specific fields (contract_id for options, etc.)

        Returns
        -------
        dict or None
            Price quotes dictionary with structure varying by position type:
            - For options: bid, ask, mid, greeks (delta, gamma, vega, theta)
            - For stocks: bid, ask, mid prices
            - For futures: settlement/mid prices
            Returns None if position data cannot be retrieved.

        Examples
        --------
        >>> position = {
        ...     'type': 'option',
        ...     'contract_id': 'SPY_230115C400',
        ...     'underlying_symbol': 'SPY'
        ... }
        >>> quotes = data.get_position_prices(position)
        >>> print(f"Option mid price: ${quotes['mid']:.2f}")
        """
        ctx = self.context

        # For estimated options, return underlying price data after updating surfaces
        if position['type'] == 'option' and (
            position.get('tracking_price_type') == 'estimated'
            or ctx.is_stochastic
        ):
            # Note: update_surfaces() would be called by Simulation class
            return ctx.price_data[position['underlying_symbol']]

        # Check if already fetched
        pos_key = constants.POS_TYPE_KEY[position['type']]
        if (
            ctx.contract_data[position['type']]
            .get(position[pos_key], {})
            .get(ctx.cur_datetime)
        ):
            return ctx.contract_data[position['type']][position[pos_key]]

        # Need to fetch - delegate to appropriate method
        data_method_map = {
            'stock': self._get_stock_data,
            'option': self._get_option_data,
            'futures_contract': self._get_futures_contract_data,
            'futures_option': self._get_futures_option_data,
        }

        if position['type'] in {'stock', 'futures_contract'}:
            return data_method_map[position['type']](position[pos_key])
        return data_method_map[position['type']](
            position[pos_key], position['underlying_symbol']
        )

    def _get_stock_data(self, underlying_symbol: str) -> dict:
        """
        Get all future stock quotes for a symbol.

        Parameters
        ----------
        underlying_symbol : str
            The symbol of the stock to get quotes for

        Returns
        -------
        dict
            The quotes for the stock
        """
        ctx = self.context
        quotes = self.client.get_stock_quotes(
            underlying_symbol=underlying_symbol,
            start_datetime=ctx.cur_datetime,
            end_datetime=ctx.end_datetime,
            time_filter=ctx.daily_timestamp
            if ctx.sim_interval == 'daily'
            else None,
        )
        ctx.contract_data['stock'][underlying_symbol] = quotes
        return quotes

    def _get_option_data(
        self, contract_id: str, underlying_symbol: str
    ) -> dict:
        """
        Get options quotes up to expiration.

        Parameters
        ----------
        contract_id : str
            The contract ID of the option to get quotes for
        underlying_symbol : str
            The underlying symbol

        Returns
        -------
        dict
            The quotes for the option
        """
        ctx = self.context
        stat_columns = [
            'mid',
            'underlying_mid',
            'implied_volatility',
        ] + constants.GREEKS
        _baseStatCols = [c for c in stat_columns if c != 'wVega']

        quotes = self.client.get_option_quotes(
            start_datetime=ctx.cur_datetime,
            columns=['quote_datetime'] + _baseStatCols,
            contract_id=contract_id,
            underlying_symbol=underlying_symbol,
        )
        ctx.contract_data['option'][contract_id] = quotes
        return quotes

    def _get_futures_contract_data(self, futures_symbol: str) -> dict:
        """
        Get futures quotes up to expiration.

        Parameters
        ----------
        futures_symbol : str
            The symbol of the futures contract to get quotes for

        Returns
        -------
        dict
            The quotes for the futures contract
        """
        ctx = self.context
        quotes = self.client.get_futures_quotes(
            futures_symbol=futures_symbol,
            start_datetime=ctx.cur_datetime,
            end_datetime=ctx.end_datetime,
            columns=['quote_datetime', 'mid', 'bid', 'ask'],
        )
        if not quotes:
            raise ValueError(f'No price data was returned for {futures_symbol}')
        ctx.contract_data['futures_contract'][futures_symbol] = quotes
        return quotes

    def _get_futures_option_data(
        self, contract_id: str, underlying_symbol: str
    ) -> dict:
        """
        Get futures options quotes up to expiration.

        Parameters
        ----------
        contract_id : str
            The contract ID of the futures option to get quotes for
        underlying_symbol : str
            The underlying symbol

        Returns
        -------
        dict
            The quotes for the futures option
        """
        ctx = self.context
        futures_greeks = [
            g for g in constants.GREEKS if g not in ['rho', 'wVega']
        ]
        _baseStatCols = [
            'settlement_price',
            'futures_close',
            'implied_volatility',
        ] + futures_greeks

        quotes = self.client.get_option_quotes(
            start_datetime=ctx.cur_datetime,
            columns=['quote_datetime'] + _baseStatCols,
            contract_id=contract_id,
            table_name='futures_options',
            underlying_symbol=underlying_symbol,
        )
        if not quotes:
            raise ValueError(
                f'No price data was returned for contract {contract_id}'
            )
        ctx.contract_data['option'][contract_id] = quotes
        return quotes

    def update_quotes(self):
        """
        Fetch and update quotes for all open positions.

        Called during simulation loop to update prices for all currently open positions.
        This is critical for tracking P&L and calculating greeks.
        """
        ctx = self.context
        for tranche in ctx.open_tranches:
            for position in tranche.positions.values():
                if position.open:
                    quotes = self.get_position_prices(position.position)
                    position.quotes = {**position.quotes, **quotes}

    # ==================== Initialization Methods ====================

    def load_stock_prices(self):
        """
        Fetch stock prices for all symbols in the simulation.

        Retrieves price data from appropriate sources based on simulation mode:
        - Historical: Queries database for actual historical prices
        - Live Trading: Fetches real-time quotes from market data providers
        - Stochastic: Generates synthetic prices using statistical models

        Called during simulation initialization to batch load all price data.
        """
        ctx = self.context

        if ctx.derivative_type == 'futures':
            ctx.price_data = {}
            return

        # Should only run for equity_index_etf
        for symbol in ctx.symbol_list:
            ctx.price_data[symbol] = {}

            if ctx.is_live_trading:
                if IB_API_AVAILABLE:
                    try:
                        ctx.price_data[symbol][ctx.datetime_list[0]] = (
                            ib_api.polygon.get_underlying_quote(symbol)
                        )
                    except Exception as e:
                        print(
                            'Failed to get live underlying price from Polygon.'
                        )
                        print(f'Error: {e}')
                else:
                    print('ib_api not available for live trading')

            elif ctx.is_stochastic:
                self._load_stochastic_prices(symbol)

            else:  # Historical data
                quotes = self.client.get_stock_quotes(
                    symbol,
                    ctx.start_datetime,
                    ctx.end_datetime,
                    ctx.daily_timestamp
                    if ctx.sim_interval == 'daily'
                    else None,
                    columns=['mid'],
                )
                for datetime, quote in quotes.items():
                    ctx.price_data[symbol][datetime] = quote['mid']

    def _load_stochastic_prices(self, symbol: str):
        """
        Load or generate stochastic price data for a symbol.

        Handles two scenarios:
        1. Pre-provided stochastic prices: Calculates drift/volatility from provided prices
        2. Generated stochastic prices: Uses stochastic module to generate synthetic prices

        Parameters
        ----------
        symbol : str
            Stock symbol to load stochastic prices for

        Notes
        -----
        Updates context with:
        - ctx.price_data[symbol]: Dictionary mapping datetime to price
        - ctx.stochastic_drift: Annualized drift rate
        - ctx.stochastic_vol: Annualized volatility

        Raises
        ------
        ModuleNotFoundError
            If stochastic module is not available when generation is needed
        """
        ctx = self.context

        if ctx.stochastic_underlying:
            # Use pre-provided stochastic prices
            prices = {
                k: v
                for (k, v) in zip(ctx.datetime_list, ctx.stochastic_underlying)
            }

            # Calculate annualized drift and volatility from provided prices
            underlying_prices = np.array(ctx.stochastic_underlying)
            returns = np.diff(underlying_prices) / underlying_prices[:-1]
            annualization_factor = (
                252 * 7 if ctx.sim_interval == 'hourly' else 252
            )
            drift = np.mean(returns) * annualization_factor
            vol = np.std(returns, ddof=1) * np.sqrt(annualization_factor)
        else:
            # Generate stochastic prices using statistical models
            if not STOCHASTIC_AVAILABLE:
                raise ModuleNotFoundError(
                    'stochastic module is required for stochastic simulations'
                )
            prices, drift, vol = stochastic.generate_stochastic_underlying(
                symbol=symbol,
                datetime_list=ctx.datetime_list,
                data_conn=self.client,
                sim_interval=ctx.sim_interval,
                daily_timestamp=ctx.daily_timestamp,
                override_drift=ctx.stochastic_drift,
                override_vol=ctx.stochastic_vol,
            )

        # Update context with prices and statistics
        ctx.price_data[symbol] = prices
        ctx.stochastic_drift = drift
        ctx.stochastic_vol = vol
        
    def _is_volatility_index(self, symbol: str) -> bool:
        """
        Determine if a symbol is a volatility index based on naming convention.
        
        Volatility indices typically follow patterns like:
        - ^VIX, ^VXN, ^VXD (CBOE volatility indices)
        - ^RVX (Russell 2000 volatility)
        - ^VXAPL, ^VXAZN, ^VXGOG (stock-specific volatility indices)
        
        Args:
            symbol: The symbol to check
            
        Returns:
            True if the symbol appears to be a volatility index, False otherwise
        """
        symbol_upper = symbol.upper().strip()
        
        # Define volatility index patterns
        # Patterns are checked in order - prefixes first, then exact matches
        volatility_patterns = [
            '^V',      # Covers most CBOE volatility indices: ^VIX, ^VXN, ^VXD, ^VXAPL, etc.
        ]
        
        # Known volatility indices that don't match the prefix patterns
        volatility_exact_matches = {
            '^RVX',    # Russell 2000 Volatility Index
        }
        
        # Check prefix patterns
        for pattern in volatility_patterns:
            if symbol_upper.startswith(pattern):
                return True
        
        # Check exact matches
        if symbol_upper in volatility_exact_matches:
            return True
        
        return False
    
    def _validate_ivp_usage(self, symbol: str, indicators: List[str]) -> None:
        """
        Validate that IVP indicators are only used on volatility indices.
        
        Raises a ValueError if IVP is applied to a non-volatility symbol.
        
        Args:
            symbol: The symbol being configured
            indicators: List of indicator specifications for this symbol
        """
        ivp_indicators = []
        for indicator in indicators:
            ind_name, _, _ = utils.parse_indicator(indicator)
            ind_name_lower = ind_name.lower()
            
            if ind_name_lower in ['ivp']:
                ivp_indicators.append(indicator)
        
        if ivp_indicators and not self._is_volatility_index(symbol):
            raise ValueError(
                f"IVP indicator cannot be applied to '{symbol}'. "
                f"IVP is only valid for volatility indices (e.g., ^VIX, ^RVX, ^VXAPL). "
                f"Found IVP indicators: {ivp_indicators}\n"
                f"If you want to analyze {symbol}'s volatility, add its corresponding "
                f"volatility index to your symbol_list and apply IVP to that index."
            )

    def load_indicator_values(self):
        """
        Calculate or fetch technical indicator values for all symbols.
        
        Processes configured technical indicators (SMA, EMA, RSI, etc.) for
        each symbol. Handles both live calculation from price data and
        historical retrieval from database.

        Called during simulation initialization to batch load all indicator data.
        """
        ctx = self.context

        if not ctx.technical_indicators:
            return

        price_data = deepcopy(ctx.price_data)

        for symbol, indicators in ctx.technical_indicators.items():
            self._validate_ivp_usage(symbol, indicators)
            if ctx.is_live_trading:
                data = technical_indicators.for_live_data(
                    symbol, indicators, ctx.cur_datetime
                )
            else:
                # Fetch past price data for indicator warmup
                max_period = max(
                    utils.parse_indicator(indicator)[1][
                        0
                    ]  # [1] is args, [0] is first arg
                    for indicator in indicators
                )
                past_datetime = (
                    pd.to_datetime(
                        ctx.start_datetime, format='%Y-%m-%d %H:%M:%S'
                    )
                    - BDay((max_period + 1))
                ).strftime('%Y-%m-%d %H:%M:%S')
                end_datetime = (
                    ctx.cur_datetime
                    if symbol in ctx.price_data
                    else ctx.end_datetime
                )
                
                quotes = None
                available_columns = list(
                    constants.AVAILABLE_QUOTE_COLUMNS.get(
                        ctx.derivative_type, frozenset()
                    )
                )
                if ctx.derivative_type == 'equity_index_etf':
                    quotes = self.client.get_stock_quotes(
                        symbol,
                        past_datetime,
                        end_datetime,
                        ctx.daily_timestamp
                        if ctx.sim_interval == 'daily'
                        else None,
                        columns=available_columns,
                    )
                elif ctx.derivative_type == 'futures':
                    quotes = self.client.get_futures_quotes(
                        symbol,
                        past_datetime,
                        end_datetime,
                        ctx.daily_timestamp
                        if ctx.sim_interval == 'daily'
                        else None,
                        columns=available_columns,
                    )

                if quotes is None:
                    print(f"Warning: No quotes found for {symbol}")
                    continue

                if symbol not in price_data:
                    price_data[symbol] = {}
              
                for dt, quote in quotes.items():
                    price_data[symbol][dt] = quote
                df = pd.DataFrame.from_dict(price_data[symbol]).T
                df.index.name = 'datetime'
                df = df.reset_index()
                data = technical_indicators.get_technical_indicators(
                    df, indicators, ctx.start_datetime
                )

            ctx.feature_data[symbol] = data
    
    def add_futures_to_price_data(self, futures_symbols: List[str]):
        """
        Add new futures contract prices to price_data.

        Parameters
        ----------
        futures_symbols : list of str
            The symbols of the futures contracts to add to the price data
        """
        ctx = self.context

        for symbol in futures_symbols:
            if symbol not in ctx.price_data:
                ctx.price_data[symbol] = {}
                quotes = self.client.get_futures_quotes(
                    symbol,
                    ctx.start_datetime,
                    ctx.end_datetime,
                    ctx.daily_timestamp
                    if ctx.sim_interval == 'daily'
                    else None,
                    columns=[default.FUTURES_PRICE_TYPE],
                )
                for datetime, quote in quotes.items():
                    ctx.price_data[symbol][datetime] = quote[
                        default.FUTURES_PRICE_TYPE
                    ]

    # ==================== Additional Lookup Methods ====================

    def get_futures_symbol(
        self,
        underlying_symbol: str,
        dte_config: Optional[dict] = None,
        datetime: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Find futures contract symbols matching specified criteria.

        Parameters
        ----------
        underlying_symbol : str
            Symbol of the underlying asset (e.g., 'ES', 'CL')
        dte_config : dict, optional
            Days-to-expiration configuration with keys:
            - 'min' : int - Minimum DTE (inclusive)
            - 'max' : int - Maximum DTE (inclusive)
            - 'target' : int - Target DTE for best match selection
            If None, returns all available futures contracts (DTE 0-1000)
        datetime : str, optional
            Quote datetime in format 'YYYY-MM-DD HH:MM:SS'
            If None, uses current simulation datetime

        Returns
        -------
        str or list or None
            - If dte_config has 'target': Returns single futures symbol string
            - If dte_config is None: Returns list of all available futures symbols
            - If no contracts found: Returns None

        Examples
        --------
        >>> # Find ES futures expiring in 20-40 days, closest to 30 days
        >>> dte_config = {'min': 20, 'max': 40, 'target': 30}
        >>> symbol = data.get_futures_symbol('ES', dte_config)
        """
        ctx = self.context

        filterCriteria = {
            'ConditionalOperator': 'AND',
            'Conditions': {
                'days_to_expiration': {
                    'ComparisonOperator': 'BETWEEN',
                    'lower': dte_config['min'] if dte_config else 0,
                    'upper': dte_config['max'] if dte_config else 1000,
                }
            },
        }

        if datetime is None:
            datetime = ctx.cur_datetime

        df = self.client.legacy_option_query(
            underlying_symbol=underlying_symbol,
            quote_datetime=datetime,
            option_type=None,
            query_args=filterCriteria,
            columns=None,
            return_pandas=True,
            data_type='futures',
        )

        if df.empty:
            return None

        return (
            list(df['futures_symbol'])
            if dte_config is None
            else [
                df.iloc[
                    (df['days_to_expiration'] - dte_config['target'])
                    .abs()
                    .idxmin()
                ]['futures_symbol']
            ]
        )

    def get_contract_price(
        self, symbol: str, contract_id: str
    ) -> Optional[dict]:
        """
        Retrieve the current price for a specific option contract.

        Parameters
        ----------
        symbol : str
            Underlying asset symbol (e.g., 'SPY', '^SPX')
        contract_id : str
            Unique identifier for the option contract

        Returns
        -------
        dict or None
            Dictionary containing price data if found:
            - 'mid' : float - Mid-market price
            Returns None if contract not found or price unavailable

        Examples
        --------
        >>> price_data = data.get_contract_price('SPY', 'SPY_230115C400')
        >>> if price_data:
        ...     print(f"Option mid price: ${price_data['mid']:.2f}")
        """
        ctx = self.context

        if not ctx.is_live_trading:
            # From RDS
            sqlQuery = f"""
            select mid
            from quotes.hourly_options FORCE INDEX (contract_datetime_index)
            where quote_datetime='{ctx.cur_datetime}'
            and contract_id='{contract_id}'
            and mid>0
            """
            df = self.client.query(sqlQuery)
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        else:
            # From Polygon
            if not IB_API_AVAILABLE:
                print('ib_api not available for live trading')
                return None

            symbol = (
                symbol
                if not symbol.startswith('^')
                else f'I:{symbol.replace("^", "")}'
            )
            contract_details = utils.parse_contract_id(contract_id)
            try:
                live_price = ib_api.polygon.get_option_stats(
                    symbol,
                    root=contract_details['root'],
                    otype=contract_details['option_type'],
                    strike=contract_details['strike'],
                    expiration=contract_details['expiration'],
                )
                return live_price
            except Exception as e:
                print(f'Failed to get contract price from Polygon: {e}')
                return None

    def get_index_value(
        self,
        index: str,
        value_type: Optional[str] = None,
        polygon_return_all: bool = False,
    ):
        """
        Retrieve the current or historical value of a market index.

        Parameters
        ----------
        index : str
            Market index identifier. Supported values:
            - 'VIX': CBOE Volatility Index
            - 'VIX3M': CBOE 3-Month Volatility Index
        value_type : str, optional
            For historical data, specifies which value to return:
            - 'close': Closing value
            - 'open': Opening value
            - 'high': High value
            - 'low': Low value
            - None: Default value based on data source
        polygon_return_all : bool, optional
            For live trading only:
            - True: Returns complete quote data dictionary
            - False: Returns only the index value (default)

        Returns
        -------
        float or dict
            - If polygon_return_all=False: Float value of the index
            - If polygon_return_all=True (live trading): Dictionary with quote data

        Raises
        ------
        ValueError
            If an unsupported index is provided

        Examples
        --------
        >>> # Get current VIX value during backtesting
        >>> vix_value = data.get_index_value('VIX')
        >>> print(f"VIX: {vix_value:.2f}")
        """
        ctx = self.context

        if index not in ['VIX', 'VIX3M']:
            raise ValueError(f'Index {index} not supported. Use VIX or VIX3M.')

        if not ctx.is_live_trading:
            index_value = starlink.supplementary.get_feature_value(
                index, ctx.cur_datetime, value_type
            )
        else:
            if not IB_API_AVAILABLE:
                print('ib_api not available for live trading')
                return None
            try:
                index_value = ib_api.polygon.get_index_quote(
                    index, return_all=polygon_return_all
                )
            except Exception as e:
                print(f'Failed to get index value from Polygon: {e}')
                return None

        return index_value

    def get_historical_iv(
        self,
        lower_dte: int,
        upper_dte: int,
        lower_delta: float,
        upper_delta: float,
        option_type: str,
        lookback_days: int = 90,
        date_time: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Retrieve historical implied volatility statistics for option contracts.

        Parameters
        ----------
        lower_dte : int
            Minimum days to expiration (inclusive)
        upper_dte : int
            Maximum days to expiration (inclusive)
        lower_delta : float
            Minimum delta value (inclusive)
        upper_delta : float
            Maximum delta value (inclusive)
        option_type : str
            Type of option contract: 'P' for puts, 'C' for calls
        lookback_days : int, optional
            Number of days to look back from reference date. Default is 90 days
        date_time : str, optional
            Reference datetime in format 'YYYY-MM-DD HH:MM:SS'
            If None, uses current simulation datetime

        Returns
        -------
        Dict[str, float]
            Dictionary containing IV statistics:
            - 'avg_iv' : float - Weighted average implied volatility
            - 'max_iv' : float - Maximum implied volatility observed
            - 'min_iv' : float - Minimum implied volatility observed
            - 'n_contracts' : float - Total number of contracts in sample
            Returns NaN values if no data found

        Examples
        --------
        >>> # Get 90-day IV stats for 30-45 DTE puts with -0.25 to -0.15 delta
        >>> iv_stats = data.get_historical_iv(
        ...     lower_dte=30, upper_dte=45,
        ...     lower_delta=-0.25, upper_delta=-0.15,
        ...     option_type='P',
        ...     lookback_days=90
        ... )
        >>> print(f"Average IV: {iv_stats['avg_iv']:.1%}")
        """

        ctx = self.context

        if not date_time:
            date_time = ctx.cur_datetime

        # Closest previous Sunday
        datetime_obj = datetime.fromisoformat(date_time)
        prev_sunday = (
            datetime_obj - timedelta(days=datetime_obj.weekday() + 1)
        ).strftime('%Y-%m-%d')

        query = f"""
            select sum(avg_iv*n_contracts)/sum(n_contracts) as avg_iv,
                max(max_iv) as max_iv,
                min(min_iv) as min_iv,
                sum(n_contracts) as n_contracts
            from quotes.historical_iv
            where model_date = '{prev_sunday}'
                and lookback_days = {lookback_days}
                and option_type = '{option_type}'
                and days_to_expiration between {lower_dte} and {upper_dte}
                and delta between {lower_delta} and {upper_delta}
        """

        ivs = self.client.query(query)

        if ivs.isna().values.any():
            return {
                'avg_iv': np.nan,
                'max_iv': np.nan,
                'min_iv': np.nan,
                'n_contracts': np.nan,
            }

        return ivs.iloc[0].to_dict()

    def close_connection(self):
        """
        Close HybridClient connection.

        Called when simulation completes. Note: When using connection pool,
        this is a no-op.
        """
        if hasattr(self.client, 'close_connection'):
            self.client.close_connection()
