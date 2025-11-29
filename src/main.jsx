import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './style.css' // lub './App.css' zalezy co masz

// UWAGA: Tutaj musi być 'app', żeby pasowało do Twojego HTML
ReactDOM.createRoot(document.getElementById('app')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
)
