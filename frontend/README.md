# PremAI Football Intelligence Platform — Frontend

The modern, dark-first React frontend for the **PremAI Football Intelligence Platform**. It provides sports analytics professionals, club scouts, and football analysts with an interactive interface for:

1. **Transfer Value Predictor**: Gradient boosted regression model estimating fair transfer market value based on player context, contract parameters, and detailed performance metrics.
2. **Match Outcome Predictor**: Pre-match 3-class probabilistic outcome forecasting (Home Win, Draw, Away Win) utilizing rolling 5-match form metrics, league standings, and historical head-to-head records.
3. **Player Scouting & Replacement**: Multidimensional player similarity search based on normalized vector cosine similarity across 10 key attacking, possession, and defensive metrics, with position-aware recruitment filters.

---

## Tech Stack & Architecture

- **Framework**: React 19 + TypeScript
- **Bundler & Dev Server**: Vite
- **Styling**: Vanilla CSS Design System (`index.css`) featuring dark sports intelligence color tokens, squircle cards, subtle borders, and responsive grid layouts.
- **Icons**: Lucide React
- **API Client**: Centralized, typed HTTP client layer (`src/api/`) with structured Pydantic error parsing.

### Directory Structure

```
frontend/src/
├── api/                   # Centralized API client modules
│   ├── client.ts          # Base fetch client, API_BASE_URL, and ApiError
│   ├── health.ts          # GET /api/health
│   ├── transfer.ts        # POST /api/transfer/predict, GET /api/transfer/model-info
│   ├── match.ts           # POST /api/match/predict, GET /api/match/model-info
│   ├── scouting.ts        # GET /api/scouting/similar, POST /api/scouting/replacements, GET /api/scouting/players
│   └── index.ts           # Re-exports
├── components/
│   ├── common/
│   │   └── ModelInfoModal.tsx    # Technical model specs modal (train/val/test splits & metrics)
│   ├── layout/
│   │   ├── Header.tsx            # Persistent branding, module pill selector, and live health badge
│   │   └── Footer.tsx            # Platform details and data source references
│   ├── transfer/
│   │   ├── TransferModule.tsx    # Module coordinator & two-column layout
│   │   ├── TransferForm.tsx      # 3 grouped form sections with Declan Rice/Haaland/Saka presets
│   │   └── TransferResult.tsx    # Valuation display card (€M) and model methodology
│   ├── match/
│   │   ├── MatchModule.tsx       # Module coordinator
│   │   ├── MatchForm.tsx         # 5 organized sections (fixture, form, league, H2H, rest) & presets
│   │   └── MatchResult.tsx       # Outcome cards, 3 probability cards, segmented probability bar
│   └── scouting/
│       ├── ScoutingModule.tsx        # Module coordinator & internal tab selector
│       ├── SimilarPlayersView.tsx    # Similarity retrieval table with Top N selector
│       ├── ReplacementSearchView.tsx # Recruitment filters (budget, age, position, minutes)
│       ├── PlayerSearchInput.tsx     # Debounced autocomplete search querying backend catalog
│       └── PlayerDetailCard.tsx      # Candidate statistical radar with 10 verified per-90 metrics
├── types/
│   └── api.ts             # TypeScript interfaces mirroring FastAPI Pydantic schemas
├── App.tsx                # Main application state and live health polling
├── main.tsx               # React root entry point
└── index.css              # Design system tokens, squircles, probability bar, and responsive rules
```

---

## Getting Started

### Prerequisites

Ensure the **FastAPI backend** is running before using the frontend:

```bash
# From workspace root
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Verify backend health at: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)  
FastAPI Interactive Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Installation

```bash
cd frontend
npm install
```

### Development Server

```bash
npm run dev
```

The application will be served at [http://localhost:5173](http://localhost:5173).

### Production Build

```bash
npm run build
```

This compiles TypeScript definitions (`tsc -b`) and generates production bundle assets in `frontend/dist/`.

---

## Environment Variables

| Variable | Description | Default Fallback |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Base URL pointing to the FastAPI backend | `http://127.0.0.1:8000` |

To override in development or staging, create a `frontend/.env` file:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Available Modules

### 1. Transfer Value Predictor
- Predicts continuous player transfer valuation in millions of Euros.
- Organizes 19 input parameters across Demographics, Contract/Club Stature, and Performance Metrics.
- Includes quick-load presets for **Declan Rice**, **Erling Haaland**, and **Bukayo Saka**.
- Displays estimated market value (`€XX.XXM`), underlying model architecture, and target transformation (`Log1p`).

### 2. Match Outcome Predictor
- Predicts pre-match probabilistic distribution across **Home Win**, **Draw**, and **Away Win**.
- Groups inputs into Fixture, 5-match rolling form, League Standing & PPG, Historical H2H records, and Rest Days.
- Includes presets for high-profile Premier League clashes (e.g. *Arsenal vs Chelsea*, *Man City vs Liverpool*).
- Displays a segmented horizontal probability bar (summing to 100%) and individual probability metric cards.

### 3. Player Scouting & Replacement
- **Similar Players**: Given a target player (e.g. *Declan Rice*, *Rodri*), computes top N closest player profiles across normalized statistical vectors.
- **Find Replacement**: Position-aware recruitment search with customizable filters for maximum valuation cap, maximum age, strict position match, and minimum minutes played.
- **Player Detail Card**: Inspects 10 key per-90 metrics (Goals/90, Assists/90, npxG/90, xAG/90, Key Passes/90, Pass Completion %, Progressive Passes/90, Progressive Carries/90, Tackles Won/90, Interceptions/90) verified from official Transfermarkt and FBref datasets.
