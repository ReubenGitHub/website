import React, { useState } from 'react';
import '../DotNetDemo.css';

export function DotNetDemo() {
    const [helloData, setHelloData] = useState(null);
    const [exampleData, setExampleData] = useState(null);
    const [echoInput, setEchoInput] = useState('');
    const [echoResponse, setEchoResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const DOTNET_API_BASE = 'http://localhost:5001/api';

    const callDotNetHello = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${DOTNET_API_BASE}/example/hello`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setHelloData(data);
        } catch (err) {
            setError(`Error calling .NET API: ${err.message}`);
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    const callDotNetData = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${DOTNET_API_BASE}/example/data`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setExampleData(data);
        } catch (err) {
            setError(`Error calling .NET API: ${err.message}`);
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    const callDotNetEcho = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${DOTNET_API_BASE}/example/echo`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: echoInput })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setEchoResponse(data);
        } catch (err) {
            setError(`Error calling .NET API: ${err.message}`);
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dotnet-demo-container">
            <h2>C# .NET Microservice Demo</h2>
            <p className="description">
                This section demonstrates communication with the C# .NET backend microservice running on port 5001.
            </p>

            {error && (
                <div className="error-message">
                    ⚠️ {error}
                </div>
            )}

            <div className="demo-section">
                <h3>1. Simple Hello Endpoint (GET)</h3>
                <button onClick={callDotNetHello} disabled={loading}>
                    {loading ? 'Loading...' : 'Call /api/example/hello'}
                </button>
                {helloData && (
                    <div className="response">
                        <h4>Response:</h4>
                        <pre>{JSON.stringify(helloData, null, 2)}</pre>
                    </div>
                )}
            </div>

            <div className="demo-section">
                <h3>2. Data Endpoint (GET)</h3>
                <button onClick={callDotNetData} disabled={loading}>
                    {loading ? 'Loading...' : 'Call /api/example/data'}
                </button>
                {exampleData && (
                    <div className="response">
                        <h4>Response:</h4>
                        <pre>{JSON.stringify(exampleData, null, 2)}</pre>
                    </div>
                )}
            </div>

            <div className="demo-section">
                <h3>3. Echo Endpoint (POST)</h3>
                <form onSubmit={callDotNetEcho}>
                    <input
                        type="text"
                        value={echoInput}
                        onChange={(e) => setEchoInput(e.target.value)}
                        placeholder="Enter a message to echo"
                        disabled={loading}
                    />
                    <button type="submit" disabled={loading || !echoInput}>
                        {loading ? 'Loading...' : 'Send Echo'}
                    </button>
                </form>
                {echoResponse && (
                    <div className="response">
                        <h4>Response:</h4>
                        <pre>{JSON.stringify(echoResponse, null, 2)}</pre>
                    </div>
                )}
            </div>

            <div className="demo-section info">
                <h4>API Base URL:</h4>
                <code>{DOTNET_API_BASE}</code>
                <h4>Available Endpoints:</h4>
                <ul>
                    <li><code>GET /api/example/hello</code> - Returns a greeting message</li>
                    <li><code>GET /api/example/data</code> - Returns sample data with items</li>
                    <li><code>POST /api/example/echo</code> - Echoes back the message sent in the request body</li>
                    <li><code>GET /health</code> - Health check endpoint for the service</li>
                </ul>
            </div>
        </div>
    );
}
