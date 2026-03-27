# 🤖 Self-Healing QA Automation Agent

A **LangChain + Playwright** powered autonomous web testing agent that reads human-language test instructions from a `.txt` file and executes them in a real Chromium browser — with real-time thought logging, self-healing selectors, and automatic test reports.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Human-language workflows** | Write tests in plain English — no Selenium locators needed |
| **GPT-4o Brain** | OpenAI Functions Agent reasons about the DOM and picks the right actions |
| **Self-Healing** | 3-strategy fallback: CSS → Aria-label → Text content |
| **Credential Memory** | Passwords/emails generated in Sign-Up are remembered for Login |
| **Mock Data Generation** | Faker-powered: emails, usernames, passwords, names, phones |
| **Rich Terminal Logging** | Colour-coded Thought / Action / Observation in real-time |
| **HTML + Text Reports** | Full run report saved to `reports/` with screenshots on failure |
| **Headless toggle** | Watch the browser live or run silently in CI |

---

## 📁 Project Structure

```
qa_agent/
├── main.py                    ← Entry point / orchestrator
├── workflow.txt               ← Your test instructions (edit this!)
├── requirements.txt
├── .env.example               ← Copy to .env and add your API key
│
├── core/
│   ├── agent_brain.py         ← LangChain OpenAI Functions Agent + callbacks
│   ├── state_manager.py       ← Shared state (credentials, page, results)
│   ├── workflow_parser.py     ← .txt → WorkflowStep list
│   ├── logger.py              ← Rich terminal logging
│   └── report_generator.py   ← HTML + text report writer
│
├── tools/
│   └── browser_tools.py       ← 11 async Playwright tools for the agent
│
└── reports/
    ├── screenshots/           ← Auto-captured on step failure
    ├── report_<ts>.txt
    └── report_<ts>.html
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+
- An OpenAI API key (GPT-4o access)

### 2. Install dependencies

```bash
cd qa_agent
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here
```

### 4. Write your workflow

Edit `workflow.txt`:

```
Step 1: Navigate to https://example.com/register
Step 2: Fill the registration form with random user data
Step 3: Click the Sign Up button
Step 4: Verify account was created successfully
Step 5: Navigate to the login page
Step 6: Login with the credentials created earlier
Step 7: Verify successful login and take a screenshot
```

### 5. Run the agent

```bash
# Visible browser (recommended for monitoring)
python main.py

# Custom workflow file
python main.py --workflow my_tests/checkout_flow.txt

# Headless (CI/CD)
python main.py --headless
```

---

## 📝 Workflow File Format

The parser is flexible — it accepts several formats:

```
## Lines starting with # are comments and are ignored

Step 1: Navigate to https://example.com       ← "Step N:" prefix
2. Click the login button                      ← "N." prefix
3) Fill the search box with "laptops"          ← "N)" prefix
Verify the results page loaded                 ← bare lines also work
```

---

## 🧰 Available Agent Tools

| Tool | Purpose |
|---|---|
| `navigate_to_url` | Go to a URL |
| `click_element` | Click with self-healing fallback |
| `type_text` | Type into inputs (clears first) |
| `select_option` | Choose from `<select>` dropdowns |
| `verify_condition` | Assert URL / text / element / title |
| `get_page_dom` | Inspect inputs, buttons, links (helps agent find selectors) |
| `generate_mock_data` | Create fake email, password, name, etc. |
| `recall_stored_value` | Retrieve previously stored credential |
| `take_screenshot` | Capture current page |
| `scroll_page` | Scroll up or down |
| `wait_seconds` | Pause between actions |

---

## 🔧 Self-Healing Logic

When a selector fails, the agent tries these in order:

1. **Primary** — your original CSS/XPath selector
2. **Aria-label** — `[aria-label*="description" i]`
3. **Placeholder** — `[placeholder*="description" i]`
4. **Text content** — `text="description"`
5. **ARIA role** — `role=button[name*="description" i]`

If all fail, the agent calls `get_page_dom` to re-inspect the live DOM and attempts a different approach before marking the step as failed.

---

## 📊 Reports

After each run, two reports are generated in `reports/`:

- **`report_<timestamp>.txt`** — plain-text summary (also printed to terminal)
- **`report_<timestamp>.html`** — rich HTML report with step details and screenshot links
- **`run.log`** — full Thought/Action/Observation log

On failure, a screenshot is automatically captured and linked in the report.

---

## 💡 Example Workflow — Sign Up + Login

```
Step 1: Navigate to https://practicetestautomation.com/practice-test-login/
Step 2: Fill in the username field with "student"  
Step 3: Fill in the password field with "Password123"
Step 4: Click the Submit button
Step 5: Verify login succeeded by checking for success message
Step 6: Take a screenshot of the logged-in page
```

---

## ⚙️ Configuration

| Env var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI key |

| CLI flag | Default | Description |
|---|---|---|
| `--workflow` | `workflow.txt` | Path to workflow file |
| `--headless` | `False` | Run without visible browser |
