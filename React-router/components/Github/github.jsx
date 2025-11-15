import React, { useState, useEffect } from 'react'

function Github() {
    const [data, setData] = useState([])
    useEffect(() => {
        fetch('https://api.github.com/users/vikalp026var')
        .then(res => res.json())
        .then(data => setData(data))
    }, [])
  return (
    <div>
        <div>Github followers: {data.followers}</div>
    <img src={data.avatar_url} alt="Github avatar" />
    <h1>{data.name}</h1>
    <p>{data.bio}</p>
    <p>{data.location}</p>
    <p>{data.blog}</p>
    <p>{data.twitter_username}</p>
    {/* <p>{data.followers}</p> */}
    <p>{data.following}</p>
    </div>
  )
}

export default Github