import React from 'react'

function InputBox({
    label,
    amount,
    onAmountChange,
    currencyOption =[],
    onCurrencyChange =[],
    selectCurrency =[],
    amountDisabled = false,
    currencyDisabled = false,
    className = "",

}) {
  return (
    <div className={`bg-white p-3 rounded-lg text-sm flex ${className}`}>
        <div className='w-1/2'>
            <label className='text-black/40 mb-2 inline-block'>{label}</label>
            <input type="text" 
            className='w-full outline-none bg-transparent relative'
            placeholder='Amount'
            value={amount}
            onChange={(e) => onAmountChange && onAmountChange(Number(e.target.value))}
            disabled={amountDisabled}
            />
        </div>
        <div className='w-1/2'>
            <p>Currency Type</p>
            <select className='w-full px-1 py-1 border-2 border-gray-300 rounded-md outline-none bg-transparent relative text-black'
            value={selectCurrency}
            onChange={(e) => onCurrencyChange && onCurrencyChange(e.target.value)}
            disabled={currencyDisabled}
            >
                {currencyOption.map((currencyOption) => (
                    <option key={currencyOption} value={currencyOption}>{currencyOption}</option>
                ))}
            </select>
        </div>
    </div>
  )
}

export default InputBox