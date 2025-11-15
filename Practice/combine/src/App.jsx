	import { useState } from 'react'
	import './App.css'

	function App() {
	const [firstNumber, setFirstNumber] = useState("")
	const [secondNumber, setSecondNumber] = useState("")
	const [result, setResult] = useState(null)

	const handleCombine = () => {
		const num1 = parseFloat(firstNumber) || 0
		const num2 = parseFloat(secondNumber) || 0
		setResult(num1 + num2)
	}

	const handleClear = () => {
		setFirstNumber("")
		setSecondNumber("")
		setResult(null)
	}

	return (
		<>
		<div className="min-h-screen bg-gradient-to-br from-purple-400 via-pink-500 to-red-500 flex items-center justify-center p-6">
			<div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl p-8 md:p-12 max-w-2xl w-full">
			<h1 className='text-5xl font-bold text-center mb-8 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent'>
				Number Combine
			</h1>
			{result !== null && (
				<div className='text-center mb-8'>
				<div className='bg-gradient-to-r from-green-100 to-blue-100 rounded-2xl p-6 border-2 border-green-200'>
					<h2 className='text-2xl font-bold text-gray-800 mb-2'>Result:</h2>
					<p className='text-4xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent'>
					{firstNumber} + {secondNumber} = {result}
					</p>
				</div>
				</div>
			)}
			<div className='space-y-8'>
				<div className='grid md:grid-cols-2 gap-6'>
				<div className='group'>
					<label className='block text-sm font-medium text-gray-700 mb-2'>First Number</label>
					<input 
					type="number" 
					placeholder='Enter first number' 
					className='w-full p-4 border-2 border-gray-200 rounded-xl text-center text-lg 
							focus:border-purple-500 focus:ring-4 focus:ring-purple-200 
							transition-all duration-300 ease-in-out
							group-hover:shadow-md
							placeholder:text-gray-400' 
					value={firstNumber}
					onChange={(e) => setFirstNumber(e.target.value)}
					/>
				</div>
				
				<div className='group'>
					<label className='block text-sm font-medium text-gray-700 mb-2'>Second Number</label>
					<input 
					type="number" 
					placeholder='Enter second number' 
					className='w-full p-4 border-2 border-gray-200 rounded-xl text-center text-lg 
							focus:border-pink-500 focus:ring-4 focus:ring-pink-200 
							transition-all duration-300 ease-in-out
							group-hover:shadow-md
							placeholder:text-gray-400' 
					value={secondNumber}
					onChange={(e) => setSecondNumber(e.target.value)}
					/>
				</div>
				</div>
				<div className='flex justify-center space-x-4 mt-8'>
				<button 
					onClick={handleCombine}
					className='bg-gradient-to-r from-purple-600 to-purple-700 
							hover:from-purple-700 hover:to-purple-800
							text-white font-semibold py-3 px-8 rounded-xl
							transition-all duration-300 ease-in-out
							transform hover:scale-105 hover:shadow-lg
							focus:outline-none focus:ring-4 focus:ring-purple-200
							disabled:opacity-50 disabled:cursor-not-allowed'
					disabled={!firstNumber && !secondNumber}
				>
					➕ Add Numbers
				</button>
				
				<button 
					onClick={handleClear}
					className='bg-gradient-to-r from-pink-500 to-pink-600 
							hover:from-pink-600 hover:to-pink-700
							text-white font-semibold py-3 px-8 rounded-xl
							transition-all duration-300 ease-in-out
							transform hover:scale-105 hover:shadow-lg
							focus:outline-none focus:ring-4 focus:ring-pink-200'
				>
					🗑️ Clear
				</button>
				</div>
			</div>
			</div>
		</div>
		</>
	)
	}

	export default App
