import { useEffect, useRef, useState } from 'react'
import './App.css'

const DEFAULT_API_URL = 'http://localhost:8000'
const COLORS = ['#22c55e', '#3b82f6', '#f97316', '#e11d48', '#a855f7']

function App() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [detections, setDetections] = useState([])
  const [imageShape, setImageShape] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [meta, setMeta] = useState(null)

  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0]
    if (!selected) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
    setDetections([])
    setMeta(null)
    setError('')
  }

  const drawOverlay = () => {
    const img = imgRef.current
    const canvas = canvasRef.current
    if (!img || !canvas || !img.complete) return

    const width = img.naturalWidth
    const height = img.naturalHeight

    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    ctx.drawImage(img, 0, 0, width, height)

    detections.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox
      const color = COLORS[idx % COLORS.length]
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      const label = `${det.class_name ?? det.class_id ?? 'defect'} (${(
        det.confidence * 100
      ).toFixed(1)}%)`

      const padding = 4
      const fontSize = 16
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`
      const textMetrics = ctx.measureText(label)
      const textHeight = fontSize + padding * 2
      const textWidth = textMetrics.width + padding * 2

      const boxX = x1
      const boxY = Math.max(y1 - textHeight, 0)

      ctx.fillStyle = `${color}dd`
      ctx.fillRect(boxX, boxY, textWidth, textHeight)
      ctx.fillStyle = '#0b0b0f'
      ctx.fillText(label, boxX + padding, boxY + textHeight - padding)
    })
  }

  useEffect(() => {
    drawOverlay()
  }, [detections, previewUrl])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Выберите изображение')
      return
    }

    setLoading(true)
    setError('')
    setDetections([])
    setMeta(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${apiUrl}/predict/image`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || `Ошибка API: ${res.status}`)
      }

      const data = await res.json()
      setDetections(data.detections ?? [])
      setImageShape(data.image_shape ?? null)
      setMeta({
        num: data.num_detections,
        time: data.processing_time_ms,
        ts: data.timestamp,
      })
    } catch (err) {
      setError(err.message || 'Не удалось получить предсказание')
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const link = document.createElement('a')
    link.download = 'prediction.jpg'
    link.href = canvas.toDataURL('image/jpeg', 0.92)
    link.click()
  }

  const handleReset = () => {
    setFile(null)
    setPreviewUrl('')
    setDetections([])
    setMeta(null)
    setError('')
    setImageShape(null)
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="eyebrow">Steel Defect Detection</p>
          <h1>Инференс модели</h1>
          <p className="muted">Загрузите изображение и получите детекции.</p>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>Запрос</h2>
          <form className="form" onSubmit={handleSubmit}>
            <label className="label" htmlFor="api-url">
              API URL
            </label>
            <input
              id="api-url"
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
            />

            <label className="label" htmlFor="file">
              Изображение (.jpg/.png)
            </label>
            <input id="file" type="file" accept="image/*" onChange={handleFileChange} />

            <div className="buttons">
              <button type="submit" disabled={loading || !file}>
                {loading ? 'Отправка...' : 'Отправить в /predict/image'}
              </button>
              <button type="button" className="ghost" onClick={handleReset}>
                Сбросить
              </button>
            </div>

            {error && <p className="error">⚠️ {error}</p>}

            {meta && (
              <div className="meta">
                <p>
                  Детекций: <strong>{meta.num}</strong>
                </p>
                <p>
                  Время: <strong>{meta.time?.toFixed?.(2) ?? meta.time} мс</strong>
                </p>
                {imageShape && (
                  <p className="muted">
                    image_shape: [{imageShape.join(', ')}]
                  </p>
                )}
              </div>
            )}
          </form>
        </section>

        <section className="panel">
          <div className="preview-header">
            <h2>Результат</h2>
            <div className="preview-actions">
              <button type="button" onClick={handleDownload} disabled={!detections.length}>
                Скачать разметку
              </button>
            </div>
          </div>

          {!previewUrl && <p className="muted">Загрузите изображение, чтобы увидеть предпросмотр.</p>}

          {previewUrl && (
            <div className="canvas-wrapper">
              {/* img нужен только чтобы получить naturalWidth/Height */}
              <img
                ref={imgRef}
                src={previewUrl}
                alt="preview"
                onLoad={drawOverlay}
                className="hidden-img"
              />
              <canvas ref={canvasRef} className="canvas" />
            </div>
          )}

          {detections.length > 0 && (
            <div className="detections">
              <h3>Детекции</h3>
              <ul>
                {detections.map((det, idx) => (
                  <li key={`${det.class_name}-${idx}`}>
                    <span className="dot" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                    {det.class_name ?? det.class_id ?? 'defect'} —{' '}
                    {(det.confidence * 100).toFixed(1)}% [{det.bbox.join(', ')}]
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
