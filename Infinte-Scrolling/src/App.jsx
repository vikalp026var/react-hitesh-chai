import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import useSearchBook from './useSearchBook'


export default function App() {
  const [query, setQuery] = useState('')
  const [pageNumber, setPageNumber] = useState(1)

  function handleSearch(e) {
    setQuery(e.target.value)
    setPageNumber(1)
  }

  const { loading, error, books, hasMore } = useSearchBook(query,pageNumber)

  function handleScroll(e) {
    if(e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight && hasMore) {
      setPageNumber(pageNumber + 1)
    }
  }

  return (
    <div onScroll={handleScroll}>
     <input type="text" onChange={handleSearch} />
     {books.map((book,index) => (
      <div key={index}>{book}</div>
     ))}
     {loading && <div>Loading...</div>}
     {error && <div>Error</div>}
    </div>
    
  )
}