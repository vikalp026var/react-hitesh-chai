import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { InputBox } from './components'
import useCurrencyInfo from './hooks/useCurrencyInfo'

function App() {
  const [amount, setAmount] = useState(0)
  const [fromCurrency, setFromCurrency] = useState("usd")
  const [toCurrency, setToCurrency] = useState("inr")
  const [convertedAmount, setConvertedAmount] = useState(0)

  const currencyInfo = useCurrencyInfo(fromCurrency)
  const currencyOption = Object.keys(currencyInfo)
  const swap = () => {
    setFromCurrency(toCurrency)
    setToCurrency(fromCurrency)
    setAmount(convertedAmount)
    setConvertedAmount(amount)
  }
  const convert = () => {
    setConvertedAmount(amount * currencyInfo[toCurrency])
  }
  const handleSubmitForm = (e) => {
    e.preventDefault()
    convert()
  }
  return (
    <>
		<div className='w-full h-screen flex flex-wrap justify-center items-center bg-gradient-to-b from-gray-400 to-gray-700 p-10'>
			<div className='w-full'>
				<div className='w-full max-w-md mx-auto bg-white rounded-lg shadow-md p-5'>
					<h1 className='text-2xl font-bold text-center mb-4'>Currency Converter</h1>
					<form onSubmit={handleSubmitForm}>
						<input type="submit" value="Convert" className='w-full bg-blue-600 text-white px-4 py-3 rounded-md cursor-pointer hover:bg-blue-700 duration-300' />
						</form>
						<div className='space-y-3'>
							<InputBox label="From" amount={amount} onAmountChange={setAmount} currencyOption={currencyOption} selectCurrency={fromCurrency} onCurrencyChange={setFromCurrency} />
							<InputBox label="To" amount={convertedAmount} onAmountChange={setConvertedAmount} currencyOption={currencyOption} selectCurrency={toCurrency} onCurrencyChange={setToCurrency} />
							<button onClick={swap} className='bg-blue-500 text-white px-4 py-2 rounded-md border-2 border-blue-500 hover:bg-blue-600 duration-200 cursor-pointer'>Swap</button>
						</div>
				</div>
			</div>
		</div>
		</>
	)
}
export default App
