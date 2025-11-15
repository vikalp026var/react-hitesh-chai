import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Home from '../components/Home/Home'
import Footer from '../components/Footer/Footer'
import Header from '../components/Header/Header'
import Layout from './Layout.jsx'
import About from './about.jsx'
import Contact from '../components/Contact/contact.jsx'
import User from '../components/User/user.jsx'
import Github from '../components/Github/github.jsx'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: '',
        element: <Home />,
      },
      {
        path: '/about',
        element: <About />,
      },
      {
        path: '/contact',
        element: <Contact />,
      },
      {
        path:'/user/:id',
        element: <User />,
      },
      {
        path:'/github',
        element: <Github />,
      }
    ],
  }
])

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
