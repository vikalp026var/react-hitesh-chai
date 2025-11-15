import React from 'react'
import UserContext from '../context/UserContext'
import { useContext } from 'react'

function Profile() {
  const {user} = useContext(UserContext);
  if(!user) return <div>please login</div>;
  return (
    <div >
        <h1 className='text-2xl text-red-500 bg-gray-300 p-10 rounded-xl'>
             Welcome {user.username}
        </h1>
    </div>
  )
}

export default Profile