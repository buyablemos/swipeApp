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

                    <div class="error">
                        <div class="error__icon">
                            <svg
                                fill="none"
                                height="24"
                                viewBox="0 0 24 24"
                                width="24"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="m13 13h-2v-6h2zm0 4h-2v-2h2zm-1-15c-1.3132 0-2.61358.25866-3.82683.7612-1.21326.50255-2.31565 1.23915-3.24424 2.16773-1.87536 1.87537-2.92893 4.41891-2.92893 7.07107 0 2.6522 1.05357 5.1957 2.92893 7.0711.92859.9286 2.03098 1.6651 3.24424 2.1677 1.21325.5025 2.51363.7612 3.82683.7612 2.6522 0 5.1957-1.0536 7.0711-2.9289 1.8753-1.8754 2.9289-4.4189 2.9289-7.0711 0-1.3132-.2587-2.61358-.7612-3.82683-.5026-1.21326-1.2391-2.31565-2.1677-3.24424-.9286-.92858-2.031-1.66518-3.2443-2.16773-1.2132-.50254-2.5136-.7612-3.8268-.7612z"
                                    fill="#393a37"
                                ></path>
                            </svg>
                        </div>
                        <div className="error__title" style={{textAlign: 'center'}}>
                            <p style={{margin: 0, lineHeight: '1.5'}}>Prawo = Ulubione ❤️</p>
                            <p style={{margin: 0, lineHeight: '1.5'}}>Góra = Koszyk 🛒</p>
                            <p style={{margin: 0, lineHeight: '1.5'}}>Lewo = Odrzucenie ❌ </p>
                        </div>

                    </div>


                    <div className="cardContainer" style={{marginTop: '55px'}}>
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
                                    style={{backgroundImage: 'url(' + item.url + ')'}}
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
                {favorites.length > 0 && (
                    <div className="list-box">
                        <h4>❤️ Ulubione ({favorites.length})</h4>
                        <ul>
                            {favorites.map((item, index) => (
                                <li key={index}>{item.name}</li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Wyświetl tylko jeśli w koszyku jest przynajmniej 1 rzecz */}
                {cart.length > 0 && (
                    <div className="list-box">
                        <h4>🛒 Koszyk ({cart.length})</h4>
                        <ul>
                            {cart.map((item, index) => (
                                <li key={index}><b>{item.name}</b> - {item.price}</li>
                            ))}
                        </ul>
                    </div>
                )}

            </div>
        </div>
    );
}

export default App;