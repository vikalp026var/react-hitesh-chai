import React, { useContext } from 'react'
import UserContext from '../context/UserContext'
import { useState } from 'react'

function Login() {
    const {setUser} = useContext(UserContext);
    const [password, setPassword] = useState("")
    const [username, setUsername] = useState("")
    const handleLogin = (e) => {
        e.preventDefault();
        setUser({username, password})
    }
  return (
    <div className='flex flex-col max-w-55 space-y-2 text-center items-center justify-center w-auto h-auto p-50 text-white bg-gray-400 rounded-xl'>
        <h1 className='text-2xl font-bold'>Login</h1>
        <input type="text" placeholder='Enter your name' onChange={(e) => setUsername(e.target.value)} className='border-2 border-gray-300 rounded-md p-2' />
        <input type="password" placeholder='Enter your password' onChange={(e) => setPassword(e.target.value)} className='border-2 border-gray-300 rounded-md p-2' />
        <button onClick={handleLogin} className='bg-blue-500 text-white px-4 py-2 rounded-md cursor-pointer'>Login</button>
    </div>
  )
}

export default Login