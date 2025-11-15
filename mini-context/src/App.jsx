import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import UserContextProvider from './context/UserContextProvider'
import Login from './components/Login'
import Profile from './components/Profile'

function App() {
  const [count, setCount] = useState(0)

  return (
    <UserContextProvider>
		<div className='flex flex-col justify-center items-center top-10 gap-10'>
			<Login/>
			<Profile/>
		</div>
    </UserContextProvider>
  )
}

export default App
