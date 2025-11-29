import React, { useState } from 'react';
import TinderCard from 'react-tinder-card';
import './App.css';


const db = [
    {
        name: 'Kurtka Jeansowa Vintage',
        price: '199 PLN',
        url: 'https://images.unsplash.com/photo-1523381210434-271e8be1f52b?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80'
    },
    {
        name: 'Czarne Sneakersy',
        price: '249 PLN',
        url: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80'
    },
    {
        name: 'Biały T-Shirt Basic',
        price: '49 PLN',
        url: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80'
    },
    {
        name: 'Wełniany Płaszcz',
        price: '450 PLN',
        url: 'https://images.unsplash.com/photo-1539533018447-63fcce2678e3?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80'
    }
];

function App() {
    const [lastDirection, setLastDirection] = useState();
    const [likedItems, setLikedItems] = useState([]);

    // Funkcja wywoływana po przesunięciu karty
    const swiped = (direction, nameToDelete) => {
        console.log('removing: ' + nameToDelete);
        setLastDirection(direction);

        // Logika biznesowa: Jeśli w prawo, dodaj do ulubionych
        if (direction === 'right') {
            setLikedItems(prev => [...prev, nameToDelete]);
        }
    };

    const outOfFrame = (name) => {
        console.log(name + ' left the screen!');
    };

    return (
        <div className="app">
            <h1>Fashion Swipe 🔥</h1>

            <div className="cardContainer">
                {db.map((character) => (
                    <TinderCard
                        className="swipe"
                        key={character.name}
                        onSwipe={(dir) => swiped(dir, character.name)}
                        onCardLeftScreen={() => outOfFrame(character.name)}
                        preventSwipe={['up', 'down']} // Blokujemy ruch góra/dół
                    >
                        <div
                            style={{ backgroundImage: 'url(' + character.url + ')' }}
                            className="card"
                        >
                            <div className="cardContent">
                                <h3>{character.name}</h3>
                                <p>{character.price}</p>
                            </div>
                        </div>
                    </TinderCard>
                ))}
            </div>

            {lastDirection && (
                <div className="info">
                    Ostatnia akcja: {lastDirection === 'right' ? '❤️ Like!' : '❌ Pass'}
                </div>
            )}

            <div className="liked-list">
                <h4>Twoje polubienia ({likedItems.length}):</h4>
                <ul>
                    {likedItems.map(item => <li key={item}>{item}</li>)}
                </ul>
            </div>
        </div>
    );
}

export default App;