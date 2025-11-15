import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [bgColor, setBgColor] = useState('')

  return (
		<div className='w-full h-screen overflow-hidden' style={{ backgroundColor: bgColor }}>
			<div className='fixed bottom-12 left-1/2 -translate-x-1/2 flex space-x-4 bg-white p-4 rounded-3xl shadow-xl'>
				<button onClick={() => setBgColor('black')} className='bg-black text-white px-4 py-2 rounded-2xl border-2 border-black hover:bg-black/80 duration-200 cursor-pointer'>
					Black
				</button>
				<button onClick={() => setBgColor('white')} className='bg-white text-black px-4 py-2 rounded-2xl border-2 border-gray-300 hover:bg-gray-100 duration-200 cursor-pointer'>
					White
				</button>
				<button onClick={() => setBgColor('#ef4444')} className='bg-red-500 text-white px-4 py-2 rounded-2xl border-2 border-red-500 hover:bg-red-600 duration-200 cursor-pointer'>
					Red
				</button>
				<button onClick={() => setBgColor('#3b82f6')} className='bg-blue-500 text-white px-4 py-2 rounded-2xl border-2 border-blue-500 hover:bg-blue-600 duration-200 cursor-pointer'>
					Blue
				</button>
				<button onClick={() => setBgColor('#22c55e')} className='bg-green-500 text-white px-4 py-2 rounded-2xl border-2 border-green-500 hover:bg-green-600 duration-200 cursor-pointer'>
					Green
				</button>
				<button onClick={() => setBgColor('#eab308')} className='bg-yellow-500 text-white px-4 py-2 rounded-2xl border-2 border-yellow-500 hover:bg-yellow-600 duration-200 cursor-pointer'>
					Yellow
				</button>
				<button onClick={() => setBgColor('#a855f7')} className='bg-purple-500 text-white px-4 py-2 rounded-2xl border-2 border-purple-500 hover:bg-purple-600 duration-200 cursor-pointer'>
					Purple
				</button>
			</div>
		</div>
    
  )
}

export default App
