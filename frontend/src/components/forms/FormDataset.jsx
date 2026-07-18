import React, {useState, useEffect} from 'react';
import './forms.css';

const DEFAULT_FILENAME = "Default: CO2 Emissions.csv"

export function FormDataset(props) {
    
    const [dataset, setDataset] = useState()
    const [datasetName, setDatasetName] = useState()
    const [datasetFields, setDatasetFields] = useState({fields: '', nonCtsFields: ''})
    const [datasetIsUpload, setDatasetIsUpload] = useState(false)
    const [count, setCount] = useState(0)
    

    const handleSubmit = () => {
        const fetchDataset = (useDefaultDataset, dataset) => {
            fetch('/api/uploadDataset', {
                method: 'post',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    useDefaultDataset,
                    ...(useDefaultDataset ? {} : { dataset }),
                    sessionId: props.sessionId
                })
            })
                .then(res => res.json())
                .then(data => {
                    if (data.datasetFields) {
                        setDatasetFields(data.datasetFields)
                    } else {
                        alert('Invalid dataset')
                    }
                })
        }
        
        if (!datasetIsUpload) {
            setDatasetName(DEFAULT_FILENAME)
            fetchDataset(true)
        } else {
            const file = dataset
            const reader = new FileReader()

            reader.onload = (e) => {
                const text = e.target.result
                const filename = file["name"]
                const maxFileSize = 2000000
            
                if (file["size"] > maxFileSize) {
                    alert('Please upload a dataset no bigger than ' + maxFileSize/1000000 + 'MB  :)')
                } else {
                    setDatasetName(filename)
                    fetchDataset(false, text)
                }
            }
        
            reader.readAsText(file)
        }
    }

    useEffect(() => { 
        if (count) {
            props.parentCallback( [datasetName, datasetFields] )
        }
        setCount(count+1);
    }, [datasetFields]);

    return (
        <form onSubmit={ (e) => {
                e.preventDefault()
                if(dataset || datasetIsUpload==false)handleSubmit()
            }}>
            <b>Select a .csv Dataset </b>
            <div className="dataset-option-group">
                <input type="radio" id="datasetDefault" name="datasetSelect" required defaultChecked onChange={(e) => {setDatasetIsUpload(false); setDataset()} } />
                <label className="dataset-option" htmlFor="datasetDefault">
                    <span className="dataset-option-icon">🚗</span>
                    <span className="dataset-option-text">
                        <span className="dataset-option-title">Default Dataset</span>
                        <span className="dataset-option-desc">Vehicle CO2 emissions data</span>
                    </span>
                </label>

                <input type="radio" id="datasetUpload" name="datasetSelect" required onChange={(e) => setDatasetIsUpload(true)} />
                <label className="dataset-option" htmlFor="datasetUpload">
                    <span className="dataset-option-icon">📁</span>
                    <span className="dataset-option-text">
                        <span className="dataset-option-title">Upload Your Own</span>
                        <span className="dataset-option-desc">CSV file, max 2MB</span>
                    </span>
                </label>
            </div>

            {datasetIsUpload && (
                <div className="glass-file-input">
                    <input type="file" accept=".csv" id="dataset" onChange={(e) => setDataset(e.target.files[0])} />
                    <label htmlFor="dataset" className="glass-file-label">
                        <span className="glass-file-icon">📄</span>
                        <span className="glass-file-text">Choose a CSV file</span>
                    </label>
                </div>
            )}

            <br></br>
            <div className="ml-button-container">
                { (!datasetIsUpload || dataset) ?
                    <button className="ml-button">Commit Dataset</button> :
                    <button disabled className="ml-button">Commit Dataset</button> }
            </div>

            {datasetName && (
                <div className="dataset-badge">
                    <span className="dataset-badge-icon">📄</span>
                    <span>Dataset selected: "{datasetName}"</span>
                    <span className="tooltip-wrapper">
                        <span className="tooltip-icon">ℹ</span>
                        <span className="tooltip-text">⚠️ WARNING: Never use data that could reveal the answer! Remove fields like 'CO2 Emissions per Cylinder' that contain the target variable. Also remove identifiers like 'City', 'Make', 'Model' that don't help predict outcomes.</span>
                    </span>
                </div>
            )}
        </form>
    )
}
