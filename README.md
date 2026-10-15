# ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score** with specific improvements, rewrite examples and keyword suggestions. Optionally paste a job description to get a keyword-match score for that role.

Built with [Streamlit](https://streamlit.io) and the Google Gemini API (Flash model).

## Features

- Reads PDF, DOCX (including tables) and TXT resumes
- Estimated ATS score out of 100, with a breakdown
- Prioritised improvements (High / Medium / Low) specific to your resume
- Before/after bullet rewrites (placeholders like `[X%]` where a metric is needed, so nothing is invented)
- Keyword suggestions, tailored to a pasted job description if you provide one
- "What the ATS sees" tab showing the raw extracted text
- Downloadable Markdown report

## How the score works

| Part | Who calculates it | What it measures |
|---|---|---|
| Format & structure | Python code (deterministic) | Contact info, standard section headings, length, bullet points, action verbs, quantified achievements, machine-readable text |
| Content quality | Gemini | Clarity, relevance, impact, consistency |
| Keyword match | Gemini (only with a job description) | Coverage of the job's skills and phrases |

Final score: `50% format + 50% content`, or with a job description `35% format + 30% content + 35% keywords`. The blend is computed in code.

> The score is an estimate for guidance. Real ATS products (Workday, Greenhouse, Lever, Taleo...) don't share a common score, so treat it as a checklist, not a verdict.

## Run locally

1. Install Python 3.10+ and get a free Gemini API key from <https://aistudio.google.com/apikey>.
2. Set up the project:

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Provide your API key. Choose one:

   - **Secrets file** (recommended). Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # GEMINI_MODEL = "gemini-3.5-flash"   # optional
     ```
   - **Environment variable:** `export GEMINI_API_KEY="your-key-here"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key-here"`)
   - **Or skip both** and paste the key into the app's sidebar.

4. Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets or environment variable | none (sidebar input is shown if missing) |
| `GEMINI_MODEL` | Streamlit secrets or environment variable | `gemini-3.5-flash` |
| `GEMINI_FALLBACK_MODELS` | Streamlit secrets or environment variable | `gemini-3.8-flash,gemini-3.1-flash-lite` |

**Overload handling:** Gemini sometimes returns `503 UNAVAILABLE` ("high demand") during traffic spikes. The app automatically retries up to 3 times with increasing waits, then falls back to the models in `GEMINI_FALLBACK_MODELS`. Key/permission errors are not retried. Set the fallbacks as a comma-separated list, for example `GEMINI_FALLBACK_MODELS = "model-a,model-b"`.

Google renames and retires models over time. If you see a "model not found" error, set `GEMINI_MODEL` to a current Flash model listed at <https://ai.google.dev/gemini-api/docs/models>.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (`app.py`, `requirements.txt`, `README.md` in the repo root).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, pick your repository and branch, and set the main file to `app.py`.
4. Open **Advanced settings**, choose a Python version (3.11 or 3.12 works well) and paste your secrets:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

**Never commit your API key.** Keep `.streamlit/secrets.toml` in `.gitignore`.

If your key is stored in the deployed app's secrets, anyone with the link can use your quota. To limit this, set the app to private in its sharing settings, set a quota/budget on the key in Google AI Studio, or leave the secret out so each visitor pastes their own key.

## Project structure

```
app.py            # the whole app: extraction, checks, Gemini call, UI
requirements.txt  # Python dependencies
README.md         # this file
```

## Privacy

Resume text is sent to Google's Gemini API for analysis. The app itself does not save uploaded files or results to disk. Check Google's Gemini API terms for how they handle API data, and avoid uploading resumes you aren't allowed to share.

## Limitations

- Scanned or image-only PDFs have no text for an ATS (or this app) to read. Export a text-based PDF or DOCX instead.
- Password-protected PDFs aren't supported.
- Text in multi-column layouts, text boxes and tables can extract in a jumbled order. The "What the ATS sees" tab helps you spot this.
- Files are limited to 5 MB; only the first ~15,000 characters of a resume are sent to the model.
- AI feedback can occasionally be wrong or generic. Check suggestions against your real experience before using them.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Gemini rejected the API key" | Re-copy the key from AI Studio; make sure there are no spaces. |
| "Gemini is overloaded right now" (503) | Temporary on Google's side. The app already retried and tried backup models; wait a minute and click Analyze again. |
| "Rate limit or quota reached" | Wait a minute, or check your quota in Google AI Studio. |
| "Could not find that model name" | Set `GEMINI_MODEL` to a current model name. |
| "Only N characters could be extracted" | The PDF is probably scanned. Export a text-based PDF/DOCX. |
| `ModuleNotFoundError` on deploy | Make sure `requirements.txt` is in the repo root and committed. |
