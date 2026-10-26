
import { useEffect, useRef, useState } from 'react'
import Papa from 'papaparse'
import './App.css'

const API_BASE = 'http://127.0.0.1:8001'

function getFilename(path) {
  if (!path) return ''
  return String(path).trim().replace(/\\/g, '/').split('/').pop()
}

function parseDictionary(value) {
  if (value && typeof value === 'object') return value
  if (!value) return {}

  try {
    return JSON.parse(value)
  } catch {
    try {
      return JSON.parse(
        String(value)
          .replace(/'/g, '"')
          .replace(/\bNone\b/g, 'null')
          .replace(/\bTrue\b/g, 'true')
          .replace(/\bFalse\b/g, 'false')
      )
    } catch {
      return {}
    }
  }
}

function getColumn(row, candidates) {
  const key = Object.keys(row).find((column) =>
    candidates.includes(column.trim().toLowerCase())
  )
  return key ? row[key] : ''
}

function normalizeApiRecord(row, index) {
  const angle = Number(row.steering_angle)

  return {
    id: row.img_id ?? index + 1,
    originalName: getFilename(row.original_name || ''),
    annotatedName: getFilename(row.annotated_name || ''),
    originalUrl: row.original_available && row.original_url
      ? `${API_BASE}${row.original_url}`
      : '',
    annotatedUrl: row.annotated_available && row.annotated_url
      ? `${API_BASE}${row.annotated_url}`
      : '',
    angle: Number.isFinite(angle) ? angle : 0,
    objects: parseDictionary(row.objects_predict),
    lanes: parseDictionary(row.lane_predict),
  }
}

function normalizeCsvRecord(row, index) {
  const originalPath = getColumn(row, [
    'img_name',
    'image',
    'image_name',
    'image_path',
    'original_image',
  ])

  const annotatedPath = getColumn(row, [
    'annotated_img',
    'annotated_image',
    'annotated_path',
    'output_image',
  ])

  const angle = Number(
    getColumn(row, ['steering_angle', 'predicted_angle', 'angle'])
  )

  return {
    id: getColumn(row, ['img_id', 'frame_id', 'id']) || index + 1,
    originalName: getFilename(originalPath),
    annotatedName: getFilename(annotatedPath),
    originalUrl: originalPath
      ? `${API_BASE}/api/images/original/${index}`
      : '',
    annotatedUrl: annotatedPath
      ? `${API_BASE}/api/images/annotated/${index}`
      : '',
    angle: Number.isFinite(angle) ? angle : 0,
    objects: parseDictionary(
      getColumn(row, [
        'objects_predict',
        'object_predictions',
        'objects',
        'detections',
      ])
    ),
    lanes: parseDictionary(
      getColumn(row, ['lane_predict', 'lane_predictions', 'lanes'])
    ),
  }
}

function normalizeClassName(value) {
  return String(value)
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
    .replace(/s$/, '')
}

function countFor(objects, aliases) {
  if (!objects || typeof objects !== 'object') return 0

  const names = aliases.map(normalizeClassName)

  const entry = Object.entries(objects).find(([key]) =>
    names.includes(normalizeClassName(key))
  )

  if (!entry) return 0

  const value = entry[1]

  if (Array.isArray(value)) return value.length

  if (value && typeof value === 'object') {
    if (Array.isArray(value.detections)) {
      return value.detections.length
    }
    if (typeof value.count === 'number') {
      return value.count
    }
    return 0
  }

  const count = Number(value)
  return Number.isFinite(count) ? count : 0
}

/* =========================
   Header and CSV upload
   ========================= */

function Header({
  total,
  fileName,
  onUpload,
  uploadRef,
  onFileChange,
  onRunInference,
  inferenceStatus,
}) {
  return (
    <header className="app-header">
      <div className="project-title">
        <h1>
          AI-Based Lane Detection and Steering Angle Prediction System
        </h1>
      </div>

      <div className="header-status">
        <span className={`status-dot ${total ? 'status-online' : ''}`} />
        {total
          ? `${total.toLocaleString()} records loaded`
          : 'No dataset'}
      </div>

      <div className="upload-area">
        <span className="selected-file" title={fileName}>
          {fileName}
        </span>

        <button
          className="upload-button"
          type="button"
          onClick={onUpload}
        >
          ↑ Upload CSV
        </button>

        <input
          ref={uploadRef}
          className="hidden-input"
          type="file"
          accept=".csv,text/csv"
          onChange={onFileChange}
        />
      </div>
    </header>
  )
}

function PanelHeading({ title, subtitle, tag }) {
  return (
    <div className="panel-heading">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {tag && <span className="panel-tag">{tag}</span>}
    </div>
  )
}

/* =========================
   Original and annotated images
   ========================= */

function ImagePanel({ title, src, filename, type }) {
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    setFailed(false)
  }, [src])

  return (
    <section className={`image-panel image-panel-${type}`}>
      <div className="image-panel-heading">
        <span className={`image-status ${type}`} />
        <span>{title}</span>

        <span className="image-filename" title={filename}>
          {filename || 'NO IMAGE PATH'}
        </span>
      </div>

      <div className="image-stage">
        {src && !failed ? (
          <img
            src={src}
            alt={title}
            onError={() => setFailed(true)}
          />
        ) : (
          <div className="image-placeholder">
            <span className="placeholder-icon">▧</span>
            <strong>
              {src ? 'IMAGE NOT FOUND' : 'IMAGE NOT AVAILABLE'}
            </strong>
            <span>
              {filename || 'Check the image path in your CSV'}
            </span>
            <small>
              Check that the Python API can find this image in the dataset
              folder.
            </small>
          </div>
        )}
      </div>
    </section>
  )
}

/* =========================
   Object detection
   ========================= */

function ObjectDetectionCard({ record }) {
  const objects = record?.objects || {}

  const items = [
    {
      label: 'Cars',
      aliases: ['car', 'cars'],
      color: 'yellow',
    },
    {
      label: 'Persons',
      aliases: ['person', 'persons', 'pedestrian', 'pedestrians'],
      color: 'blue',
    },
    {
      label: 'Trucks',
      aliases: ['truck', 'trucks'],
      color: 'red',
    },
    {
      label: 'Traffic lights',
      aliases: ['traffic_light', 'traffic_lights', 'trafficlight'],
      color: 'purple',
    },
    {
      label: 'Bus',
      aliases: ['bus', 'buses'],
      color: 'blue',
    },
    {
      label: 'Stop sign',
      aliases: ['stop_sign', 'stopsign', 'stop sign'],
      color: 'red',
    },
    {
      label: 'Bicycle',
      aliases: ['bicycle', 'bicycles', 'bike', 'bikes'],
      color: 'yellow',
    },
    {
      label: 'Motorbike',
      aliases: ['motorbike', 'motorbikes', 'motorcycle', 'motorcycles'],
      color: 'purple',
    },
  ]

  return (
    <section className="side-card objects-card">
      <PanelHeading
        title="Object Detection"
        subtitle="Objects in the selected frame"
        tag="VISION"
      />

      <div className="object-grid">
        {items.map((item) => (
          <div className="object-item" key={item.label}>
            <span className={`object-indicator ${item.color}`} />
            <span className="object-label">{item.label}</span>
            <strong>{countFor(objects, item.aliases)}</strong>
          </div>
        ))}
      </div>

      <div className="object-card-footer">
        <span>FRAME ID: {record?.id ?? '—'}</span>
        <span>{items.length} CATEGORIES</span>
      </div>
    </section>
  )
}

/* =========================
   Steering analysis
   ========================= */

function SteeringWheelCard({ angle }) {
  const safeAngle = Math.max(-180, Math.min(180, angle))

  const markerPosition = Math.max(
    0,
    Math.min(100, ((angle + 45) / 90) * 100)
  )

  const direction =
    angle > 1
      ? 'TURNING RIGHT'
      : angle < -1
        ? 'TURNING LEFT'
        : 'NEAR STRAIGHT'

  return (
    <section className="side-card steering-card">
      <PanelHeading
        title="Steering Analysis"
        subtitle="Predicted steering angle"
        tag="CNN OUTPUT"
      />

      <div className="steering-content">
        <div className="wheel-container">
          <svg
            className="steering-svg"
            viewBox="0 0 200 200"
            style={{ transform: `rotate(${safeAngle}deg)` }}
            role="img"
            aria-label={`Steering angle ${angle.toFixed(2)} degrees`}
          >
            <circle
              cx="100"
              cy="100"
              r="75"
              fill="none"
              stroke="#596779"
              strokeWidth="17"
            />

            <circle
              cx="100"
              cy="100"
              r="75"
              fill="none"
              stroke="#1B2636"
              strokeWidth="6"
            />

            <path
              d="M48 64 Q100 32 152 64"
              fill="none"
              stroke="#AAB7C8"
              strokeWidth="7"
              strokeLinecap="round"
            />

            <path
              d="M48 64 L63 112 L88 112 L100 93 L112 112 L137 112 L152 64"
              fill="none"
              stroke="#8492A5"
              strokeWidth="13"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            <path
              d="M88 112 L76 157 M112 112 L124 157"
              fill="none"
              stroke="#68788D"
              strokeWidth="13"
              strokeLinecap="round"
            />

            <circle
              cx="100"
              cy="100"
              r="25"
              fill="#182333"
              stroke="#9AA9BB"
              strokeWidth="3"
            />

            <circle
              cx="100"
              cy="100"
              r="14"
              fill="#080E19"
              stroke="#53657B"
              strokeWidth="2"
            />

            <circle cx="100" cy="100" r="5" fill="#4DE0BE" />
          </svg>
        </div>

        <div className="steering-readout">
          <span className="readout-label">STEERING ANGLE</span>

          <strong className="angle-number">
            {angle > 0 ? '+' : ''}
            {angle.toFixed(2)}
            <small>°</small>
          </strong>

          <span
            className={`direction-tag ${
              angle > 1 ? 'right-turn' : angle < -1 ? 'left-turn' : ''
            }`}
          >
            {direction}
          </span>

          <div className="angle-scale">
            <span>-45°</span>
            <span>0°</span>
            <span>+45°</span>
          </div>

          <div className="angle-track">
            <span
              className="angle-marker"
              style={{ left: `${markerPosition}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  )
}

/* =========================
   Frame playback
   ========================= */

function PlaybackCard({
  frame,
  total,
  playing,
  setPlaying,
  setFrame,
}) {
  return (
    <section className="side-card playback-card">
      <PanelHeading
        title="Frame Playback"
        subtitle="Navigate dataset frames"
        tag={`${total.toLocaleString()} FRAMES`}
      />

      <div className="playback-controls">
        <button
          type="button"
          aria-label="First frame"
          disabled={frame <= 1 || !total}
          onClick={() => setFrame(1)}
        >
          |◀
        </button>

        <button
          type="button"
          aria-label="Previous frame"
          disabled={frame <= 1 || !total}
          onClick={() => setFrame((value) => Math.max(1, value - 1))}
        >
          ◀
        </button>

        <button
          type="button"
          className="play-main"
          disabled={!total}
          onClick={() => {
            if (playing && frame >= total) {
              setFrame(1)
              setPlaying(true)
            } else {
              setPlaying((value) => !value)
            }
          }}
        >
          {playing ? 'Ⅱ' : '▶'}
        </button>

        <button
          type="button"
          aria-label="Next frame"
          disabled={frame >= total || !total}
          onClick={() => setFrame((value) => Math.min(total, value + 1))}
        >
          ▶
        </button>

        <button
          type="button"
          aria-label="Last frame"
          disabled={frame >= total || !total}
          onClick={() => {
            setFrame(total)
            setPlaying(false)
          }}
        >
          ▶|
        </button>
      </div>

      <input
        className="playback-range"
        type="range"
        min="1"
        max={Math.max(total, 1)}
        value={Math.min(frame, Math.max(total, 1))}
        disabled={!total}
        onChange={(event) => {
          setPlaying(false)
          setFrame(Number(event.target.value))
        }}
        aria-label="Select frame"
      />

      <div className="playback-footer">
        <span>FRAME {String(frame).padStart(3, '0')}</span>
        <span>
          {total ? `${Math.round((frame / total) * 100)}%` : '0%'}
        </span>
      </div>
    </section>
  )
}

/* =========================
   Main application
   ========================= */

function App() {
  const [records, setRecords] = useState([])
  const [frame, setFrame] = useState(1)
  const [playing, setPlaying] = useState(false)
  const [fileName, setFileName] = useState('predictions.csv')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const uploadRef = useRef(null)

  /* Load the existing dataset from the Python API */

  useEffect(() => {
    let cancelled = false

    async function loadDefaultFrames() {
      setLoading(true)
      setError('')

      try {
        const response = await fetch(`${API_BASE}/api/frames`)

        if (!response.ok) {
          throw new Error(`Python API returned HTTP ${response.status}`)
        }

        const payload = await response.json()

        if (!Array.isArray(payload.frames)) {
          throw new Error('The API response does not contain a frames list.')
        }

        if (!cancelled) {
          setRecords(payload.frames.map(normalizeApiRecord))
          setFrame(1)
          setFileName('predictions.csv')
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            `Could not load the Python API at ${API_BASE}. ${loadError.message}`
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    loadDefaultFrames()

    return () => {
      cancelled = true
    }
  }, [])

  /* Upload and parse a CSV file */

  function loadCsvFile(file) {
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a CSV file.')
      return
    }

    setPlaying(false)
    setLoading(true)
    setError('')
    setFileName(file.name)

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header) => header.trim().replace(/^\uFEFF/, ''),

      complete: (result) => {
        const rows = result.data.filter((row) =>
          Object.values(row).some((value) => String(value ?? '').trim())
        )

        if (!rows.length) {
          setRecords([])
          setError('The selected CSV contains no data rows.')
          setLoading(false)
          return
        }

        const headers = Object.keys(rows[0]).map((header) =>
          header.trim().toLowerCase()
        )

        const requiredColumnGroups = [
          ['img_name', 'image', 'image_name', 'image_path', 'original_image'],
          ['annotated_img', 'annotated_image', 'annotated_path', 'output_image'],
          ['steering_angle', 'predicted_angle', 'angle'],
        ]

        const missingColumns = requiredColumnGroups.some((group) =>
          !group.some((candidate) => headers.includes(candidate))
        )

        if (missingColumns) {
          setRecords([])
          setError(
            'CSV must contain image path, annotated image path, and steering angle columns.'
          )
          setLoading(false)
          return
        }

        const parsed = rows.map(normalizeCsvRecord)

        setRecords(parsed)
        setFrame(1)
        setLoading(false)

        if (result.errors.length) {
          console.warn('CSV parsing warnings:', result.errors)
        }
      },

      error: (parseError) => {
        setRecords([])
        setError(parseError.message || 'Could not read the CSV file.')
        setLoading(false)
      },
    })
  }

  /* Play frames sequentially */

  useEffect(() => {
    if (!playing || !records.length) return undefined

    const timer = setInterval(() => {
      setFrame((current) => {
        if (current >= records.length) {
          setPlaying(false)
          return records.length
        }

        return current + 1
      })
    }, 75)

    return () => clearInterval(timer)
  }, [playing, records.length])

  const record = records[frame - 1] ?? null

  return (
    <main className="app-shell">
      <Header
        total={records.length}
        fileName={fileName}
        onUpload={() => uploadRef.current?.click()}
        uploadRef={uploadRef}
        onFileChange={(event) => {
          loadCsvFile(event.target.files?.[0])
          event.target.value = ''
        }}
      />

      {error && <div className="error-banner">{error}</div>}
      {loading && <div className="loading-banner">LOADING DATASET…</div>}

      <section className="workspace">
        <div className="image-column">
          <div className="workspace-heading">
            <div>
              <h2>Road Scene Monitoring</h2>
              <p>Original image and annotated output</p>
            </div>

            <span className="frame-pill">
              FRAME ID / {record?.id ?? '—'}
            </span>
          </div>

          <ImagePanel
            title="01 / ORIGINAL IMAGE"
            src={record?.originalUrl}
            filename={record?.originalName}
            type="original"
          />

          <ImagePanel
            title="02 / ANNOTATED IMAGE"
            src={record?.annotatedUrl}
            filename={record?.annotatedName}
            type="annotated"
          />
        </div>

        <aside className="analysis-column">
          <ObjectDetectionCard record={record} />

          <SteeringWheelCard angle={record?.angle ?? 0} />

          <PlaybackCard
            frame={frame}
            total={records.length}
            playing={playing}
            setPlaying={setPlaying}
            setFrame={setFrame}
          />
        </aside>
      </section>
    </main>
  )
}

export default App
