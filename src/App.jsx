import React, { useState, useEffect } from 'react';
import './App.css';
import logoKocour from './assets/kocour.png'; // Upewnij się, że ścieżka jest poprawna

function App() {
    // --- STANY DANYCH (z App.jsx) ---
    const [products, setProducts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [favorites, setFavorites] = useState([]);
    const [cart, setCart] = useState([]);

    // --- STANY ANIMACJI (z App2.jsx) ---
    const [lastDirection, setLastDirection] = useState(null);
    const [animatingCardId, setAnimatingCardId] = useState(null); // Używamy ID zamiast Name dla pewności
    const [animationDirection, setAnimationDirection] = useState(null);
    const [isAnimating, setIsAnimating] = useState(false);

    const ANIMATION_DURATION = 600; // Czas trwania animacji w ms

    // --- POBIERANIE DANYCH ---
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

    // Sprawdzamy, czy można wykonać swipe (czy są karty i czy nic się teraz nie animuje)
    const canSwipe = products.length > 0 && !isAnimating;

    // --- LOGIKA SWIPE (Manualna animacja z App2) ---
    const swipe = (dir) => {
        if (!canSwipe) return;

        // Bierzemy ostatnią kartę z listy (tę na samej górze stosu)
        const currentCard = products[products.length - 1];

        // 1. Uruchom animację
        setIsAnimating(true);
        setAnimatingCardId(currentCard.id);
        setAnimationDirection(dir);
        setLastDirection(dir);

        // 2. Czekamy aż animacja się skończy, a potem aktualizujemy dane
        setTimeout(() => {
            // Usuwamy kartę z listy (zamiast outOfFrame)
            setProducts(prev => prev.slice(0, -1));

            // Logika biznesowa (Koszyk / Ulubione)
            if (dir === 'right') {
                setFavorites(prev => [...prev, currentCard]);
            } else if (dir === 'up') {
                setCart(prev => [...prev, currentCard]);
            }

            // Reset stanów animacji
            setAnimatingCardId(null);
            setAnimationDirection(null);
            setIsAnimating(false);
        }, ANIMATION_DURATION);
    };

    // --- OBSŁUGA KLAWIATURY ---
    useEffect(() => {
        const handleKeyDown = (event) => {
            if (!canSwipe) return;
            switch (event.key) {
                case 'ArrowLeft': swipe('left'); break;
                case 'ArrowRight': swipe('right'); break;
                case 'ArrowUp': swipe('up'); break;
                default: break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [canSwipe, products]); // Zależności, aby mieć dostęp do aktualnego stanu

    // Pomocnicza funkcja do klas CSS
    const getCardClassName = (id) => {
        let className = 'card-wrapper';
        if (animatingCardId === id) {
            className += ` animating-${animationDirection}`;
        }
        return className;
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



            {/* KONTENER KART */}
            {products.length > 0 ? (
                <div className="cardContainer" style={{marginTop: '20px', position: 'relative'}}>
                    {products.map((item, index) => (
                        <div
                            key={item.id}
                            className={getCardClassName(item.id)}
                            style={{ zIndex: index }} // Ważne: ostatni element na górze
                        >
                            <div
                                style={{backgroundImage: 'url(' + item.url + ')'}}
                                className="card"
                            >
                                {/* Overlay (nakładka z ikoną podczas animacji) */}
                                {animatingCardId === item.id && (
                                    <div className={`swipe-overlay ${animationDirection}`}>
                                        {animationDirection === 'left' && <span>❌</span>}
                                        {animationDirection === 'right' && <span>❤️</span>}
                                        {animationDirection === 'up' && <span>🛒</span>}
                                    </div>
                                )}

                                <div className="cardContent">
                                    <h3>{item.name}</h3>
                                    <p>{item.price}</p>
                                </div>

                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="empty-state">
                    <h2>To już wszystko! 🤷‍♂️</h2>
                    <button onClick={() => window.location.reload()} style={{padding: '10px 20px', fontSize: '16px', cursor: 'pointer'}}>
                        Załaduj ponownie
                    </button>
                </div>
            )}

            {/* PRZYCISKI STEROWANIA */}
            {products.length > 0 && (
                <div className="buttons" style={ {display: 'flex', gap: '25px', justifyContent: 'center'}}>
                    <button className="btn btn-left" onClick={() => swipe('left')} disabled={!canSwipe}>❌</button>
                    <button className="btn btn-up" onClick={() => swipe('up')} disabled={!canSwipe}>🛒</button>
                    <button className="btn btn-right" onClick={() => swipe('right')} disabled={!canSwipe}>❤️</button>
                </div>
            )}

            {/* INFO O OSTATNIEJ AKCJI */}
            {lastDirection && products.length > 0 && (
                <div className={`info direction-${lastDirection}`} style={{textAlign: 'center', marginTop: '20px', fontWeight: 'bold'}}>
                    {lastDirection === 'right' ? '❤️ Dodano do ulubionych' :
                        lastDirection === 'up' ? '🛒 Dodano do koszyka' :
                            '❌ Pass'}
                </div>
            )}
            {/* ERROR / INSTRUKCJA */}
            {products.length > 0 && (
                <div className="error">
                    <div className="error__icon">
                        {/* Ikona SVG */}
                        <svg fill="none" height="24" viewBox="0 0 24 24" width="24" xmlns="http://www.w3.org/2000/svg">
                            <path d="m13 13h-2v-6h2zm0 4h-2v-2h2zm-1-15c-1.3132 0-2.61358.25866-3.82683.7612-1.21326.50255-2.31565 1.23915-3.24424 2.16773-1.87536 1.87537-2.92893 4.41891-2.92893 7.07107 0 2.6522 1.05357 5.1957 2.92893 7.0711.92859.9286 2.03098 1.6651 3.24424 2.1677 1.21325.5025 2.51363.7612 3.82683.7612 2.6522 0 5.1957-1.0536 7.0711-2.9289 1.8753-1.8754 2.9289-4.4189 2.9289-7.0711 0-1.3132-.2587-2.61358-.7612-3.82683-.5026-1.21326-1.2391-2.31565-2.1677-3.24424-.9286-.92858-2.031-1.66518-3.2443-2.16773-1.2132-.50254-2.5136-.7612-3.8268-.7612z" fill="#393a37"></path>
                        </svg>
                    </div>
                    <div className="error__title" style={{textAlign: 'center'}}>
                        <p style={{margin: 0, lineHeight: '1.5'}}>Prawo  = Ulubione ❤️</p>
                        <p style={{margin: 0, lineHeight: '1.5'}}>Góra  = Koszyk 🛒</p>
                        <p style={{margin: 0, lineHeight: '1.5'}}>Lewo  = Odrzucenie ❌</p>
                    </div>
                </div>
            )}
            {/* LISTY (ULUBIONE / KOSZYK) - Wersja Premium */}
            <div className="lists-container">

                {/* ULUBIONE */}
                {favorites.length > 0 && (
                    <div className="list-col">
                        <div className="list-header header-fav">
                            <h4>❤️ Ulubione <span className="badge">{favorites.length}</span></h4>
                        </div>
                        <div className="list-scroll">
                            {favorites.map((item, index) => (
                                <div key={index} className="list-item">
                                    <img src={item.url} alt="mini" className="list-thumb" />
                                    <div className="list-details">
                                        <span className="list-name">{item.name}</span>
                                        <span className="list-price">{item.price}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* KOSZYK */}
                {cart.length > 0 && (
                    <div className="list-col">
                        <div className="list-header header-cart">
                            <h4>🛒 Koszyk <span className="badge">{cart.length}</span></h4>
                        </div>
                        <div className="list-scroll">
                            {cart.map((item, index) => (
                                <div key={index} className="list-item">
                                    <img src={item.url} alt="mini" className="list-thumb" />
                                    <div className="list-details">
                                        <span className="list-name">{item.name}</span>
                                        <span className="list-price">{item.price}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>


        </div>
    );
}

export default App;