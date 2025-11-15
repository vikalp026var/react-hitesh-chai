import {useEffect, useState} from 'react'


export default function useCurrencyInfo(currency) {
    const [currencyInfo, setCurrencyInfo] = useState({})
    useEffect(() => {
        fetch(`https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/${currency}.json`)
        .then(res => res.json())
        .then(data => setCurrencyInfo(data[currency]))
    },[currency])
    return currencyInfo
}