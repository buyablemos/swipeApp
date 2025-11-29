import React, { useState, useEffect } from 'react';
import TinderCard from 'react-tinder-card';
import './App.css';


import logoKocour from './assets/kocour.png';

function App() {
    const [lastDirection, setLastDirection] = useState();
    const [favorites, setFavorites] = useState([]);
    const [cart, setCart] = useState([]);

    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);

    // Pobieranie danych
    useEffect(() => {
        fetch('https://fakestoreapi.com/products/')
            .then(res => res.json())
            .then(json => {
                const formattedData = json.map(item => ({
                    id: item.id,
                    name: item.title.substring(0, 20) + '...',
                    price: item.price + ' USD',
                    url: item.image
                }));
                setProducts(formattedData);
                setLoading(false);
            })
            .catch(err => console.error("Błąd:", err));
    }, []);

    const swiped = (direction, item) => {
        setLastDirection(direction);
        if (direction === 'right') {
            setFavorites(prev => [...prev, item]);
        } else if (direction === 'up') {
            setCart(prev => [...prev, item]);
        }
    };

    // --- TO JEST NOWOŚĆ: USUWANIE ---
    const outOfFrame = (idToRemove) => {
        console.log('Usuwam z pamięci ID:', idToRemove);
        // Filtrujemy listę i zostawiamy tylko te produkty, które NIE mają tego ID
        setProducts(currentProducts => currentProducts.filter(p => p.id !== idToRemove));
    };

    if (loading) {
        return <div className="app"><h1>Ładowanie ubrań... ⏳</h1></div>;
    }

    return (
        <div className="app">
            <h1 className="shop-header">
                Kocour shop
                <img src={logoKocour} alt="Logo kocour" className="header-logo"/>
            </h1>

            {/* Wyświetlamy to tylko, jeśli są jeszcze produkty */}
            {products.length > 0 ? (
                <>
                <p style={{color: '#777', fontSize: '14px', marginTop: '-15px'}}>
                        Prawo = Ulubione ❤️ | Góra = Koszyk 🛒
                    </p>

                    <div className="cardContainer">
                        {products.map((item) => (
                            <TinderCard
                                className="swipe"
                                key={item.id}
                                onSwipe={(dir) => swiped(dir, item)}
                                // Tutaj wywołujemy usuwanie po zakończeniu animacji
                                onCardLeftScreen={() => outOfFrame(item.id)}
                                preventSwipe={['down']}
                            >
                                <div
                                    style={{ backgroundImage: 'url(' + item.url + ')' }}
                                    className="card"
                                >
                                    <div className="cardContent">
                                        <h3>{item.name}</h3>
                                        <p>{item.price}</p>
                                    </div>
                                </div>
                            </TinderCard>
                        ))}
                    </div>
                </>
            ) : (
                // Co pokazać, gdy usuniemy wszystkie karty?
                <div className="empty-state">
                    <h2>To już wszystko! 🤷‍♂️</h2>
                    <button onClick={() => window.location.reload()} style={{padding: '10px 20px', fontSize: '16px', cursor: 'pointer'}}>
                        Załaduj ponownie
                    </button>
                </div>
            )}

            {lastDirection && products.length > 0 && (
                <div className="info">
                    {lastDirection === 'right' ? '❤️ Dodano do ulubionych' :
                        lastDirection === 'up' ? '🛒 SUPERLIKE! W koszyku' :
                            '❌ Pass'}
                </div>
            )}

            <div className="lists-container">
                <div className="list-box">
                    <h4>❤️ Ulubione ({favorites.length})</h4>
                    <ul>
                        {favorites.map((item, index) => (
                            <li key={index}>{item.name}</li>
                        ))}
                    </ul>
                </div>
                <div className="list-box">
                    <h4>🛒 Koszyk ({cart.length})</h4>
                    <ul>
                        {cart.map((item, index) => (
                            <li key={index}><b>{item.name}</b> - {item.price}</li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
}

export default App;