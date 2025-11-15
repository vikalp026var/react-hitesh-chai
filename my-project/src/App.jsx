import { useState, useCallback, useEffect } from 'react'
import { toast } from 'react-toastify'
import './App.css'

function App() {
  const [length, setLength] = useState(8);
  const [numberAllowed, setNumberAllowed] = useState(false);
  const [charAllowed, setCharAllowed] = useState(false);
  const [password, setPassword] = useState("");
  
  const passwordGenerator = useCallback(() => {
	let pass = ""
	let str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	if(numberAllowed){
		str += "0123456789"
	}
	if(charAllowed) str += "!@#$%^&*_+=[]{}~`"
	for(let i=0; i < length; i++){
        let char = Math.floor(Math.random() * str.length + 1)
		pass += str.charAt(char)
	}
	setPassword(pass)
  		} , [length, numberAllowed, charAllowed, setPassword]
	)

  useEffect(() => {
	passwordGenerator()
  }, [length, numberAllowed, charAllowed, passwordGenerator])
     
  const handleCopy = useCallback(
	 () => {
		navigator.clipboard.writeText(password)
		toast.success("Password copied to clipboard")
	 }, [password]
  )
  return (
    <div>
		<div className='relative mx-auto px-10 py-12 bg-gray-700 shadow-md  max-w-xl rounded-xl text-orange-500 flex flex-col mt-10'>
			<div className='flex flex-row rounded-lg overflow-hidden mb-4'>
				<input type="text" value={password} className='outline-none bg-white w-full py-1 px-3 border border-white' placeholder='password' readOnly/>
				<button className='outline-none bg-blue-500 py-2 px-3 cursor-pointer hover:bg-blue-700 duration-400'
				onClick={handleCopy}
				>copy</button>
			</div>
			<div className='flex text-sm gap-x-2'>
               <div className='flex items-center gap-x-2'>
				   <input type="range" min={8} max={15} value={length} className='cursor-pointer' onChange={(e) => setLength(e.target.value)}/>
				   <label htmlFor="length">Length: {length}</label>
				   <input type="checkbox" checked={numberAllowed} onChange={() => setNumberAllowed(!numberAllowed)} className='cursor-pointer'/>
				   <label htmlFor="numberAllowed">Number</label>
				   <input type="checkbox" checked={charAllowed} onChange={() => setCharAllowed(!charAllowed)} className='cursor-pointer'/>
				   <label htmlFor="charAllowed">Character</label>
			   </div>
			</div>
		</div>
    </div>
  )
}

export default App
