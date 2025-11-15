import { useState } from 'react'
import './App.css'

function App() {
  const [counter, setCounter] = useState(0);

  const increment = () => {
    setCounter(prevCounter => prevCounter + 1);
    if(counter >= 10) {
      setCounter(10);
    }
  }

  const decrement = () => {
    setCounter(counter - 1);
    if(counter <= 0) {
      setCounter(0);
    }
  }
  return (
    <>
      <h1>Chai aur React</h1>
      <h2>Counter Value: {counter}</h2>
      <button onClick = {increment} >Increment</button>
      <button onClick = {decrement} >Decrement</button>
    </>
  )
}

export default App
