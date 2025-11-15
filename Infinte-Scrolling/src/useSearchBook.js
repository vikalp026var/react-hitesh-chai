import React from 'react'
import { useState, useEffect } from 'react';

function useSearchBook(query, pageNumber) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [books, setBooks] = useState([]);
    const [hasMore, setHasMore] = useState(true);
    useEffect(() => {
        setLoading(true);
        setError(null);
        setHasMore(true);
        fetch(`https://openlibrary.org/search.json?q=${query}&page=${pageNumber}`)
        .then(res => res.json())
        .then(data => {
            setBooks(prevBooks => [...new Set([...prevBooks, ...data.docs.map(book => book.title)])]);
            setLoading(false);
            setError(null);
            setHasMore(data.docs.length > 0);
        })
        .catch(err => {
            setError(err);
            setLoading(false);
            setHasMore(false);
        })
    },[query,pageNumber,loading,error,hasMore])
  return { loading, error, books, hasMore };
}

export default useSearchBook