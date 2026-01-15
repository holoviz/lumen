# :material-map: Navigating the Lumen UI

Lumen Explorer combines data exploration and AI-powered analysis in one split-screen interface. This guide shows you where everything is and how to use it.

## Get started

### Your interface at a glance

When you open Lumen Explorer, you'll see the chat panel where you ask questions and a consistent left sidebar for switching between **Explore** and **Report** mode. The results area appears once you start analyzing, and the exploration navigation menu appears as soon as you have explorations.

**Splash tabs** — The launch screen includes **Chat with data** (ask questions) and **Select data to explore** (pick a table and start an exploration).

**Chat panel** — Ask questions about your data in plain English. The AI will generate queries and visualizations automatically.

**Results area** — View interactive tables, charts, and analysis results. Use Graphic Walker to filter, sort, and explore visually. Appears when you ask your first question.

**Navigation menu** (left panel next to the main content) — Track all your analyses in a tree. Each exploration is a separate session, so you can branch off previous results, nest follow-ups, and compare different approaches.

### Ask questions

You can begin interrogating your data in three ways:

**Type a direct question** — Enter any question in the "Ask anything..." text box. For example: "What are the top 5 customers by revenue?" or "Show me sales trends over time." Hit enter or click the send arrow.

**Use quick action buttons** — Below the text box, you'll find preset buttons that handle common tasks without typing:

- **What data is available?** — Get an instant summary of your dataset's structure and contents.
- **Show me a dataset** — Display a dataset and its columns.
- **Visualize the data** — Generate an automatic visualization to explore patterns and distributions.
- **Show a demo** — See a pre-built example of what Lumen can do (appears when demo mode is enabled).

**Start from a table** — On the splash screen, switch to **Select data to explore**, choose a table, and click **Explore** to launch a new exploration for that dataset.

### Interact with messages

The chat interface provides several built-in actions:

**Undo** — Remove the last message and its response.

**Rerun** — Re-execute the last query to regenerate results.

**Clear** — Delete all messages and start a new conversation.

**Upload files** — You can attach files directly in the chat input to add new data sources or provide additional context.

## Work with results

### Control how analysis runs

Use **Settings** in the left sidebar to toggle three analysis features and configure your LLM:

**Chain of Thought** — When enabled, the AI shows its reasoning steps. Use this when you want to understand *how* the AI arrived at an answer. Disabled by default.

**SQL Planning** — When enabled, the AI plans its SQL query before executing it. This improves accuracy for complex questions. Enabled by default.

**Validation Step** — When enabled, the AI double-checks results for correctness. This catches data errors early. Enabled by default.

Simply click the toggle next to each option to turn it on or off.

### Navigate the sidebar

The left sidebar provides quick access to all major features:

**Explore** — The main chat and analysis mode (default view).

**Report** — Switch to report mode to see all explorations in a consolidated view.

**Sources** — Add new data sources (CSV, Parquet, JSON, etc.) or view connected sources.

**Settings** — Toggle Chain of Thought, SQL Planning, and Validation Step options, and configure your LLM.

**Help** — Open quick help for the interface, explorations, results, and export.

### Organize your explorations

Each time you run a SQL query that returns data, a new **exploration** is created. Explorations are persistent, contextual workspaces for working with a specific dataset. They capture the conversation, generated analyses, visualizations, and other data artifacts so they can evolve over time and be revisited or exported. The navigation menu shows them as a persistent tree across both Explore and Report mode.

**Home** — Your starting point. All initial questions begin here.

**Individual explorations** — Each SQL query result gets its own exploration with a descriptive title (e.g., "Top Customers by Revenue"). Select a node in the navigation menu to switch.

**Follow-up explorations** — If a question builds on a previous result, it appears nested under the parent exploration.

**Remove explorations** — Use the item menu (⋮) in the navigation menu and choose **Remove** to delete an exploration.

### View results

When an analysis completes, you'll see:

**Result tabs** — Each analysis output is shown as a tab (for example **Data Source** or a query result). Select tabs to switch outputs.

**Editor + output** — The editor appears above the output so you can review or tweak the query/spec while seeing results update below.

**Overview tab** — An interactive table powered by Graphic Walker. Here you can sort, filter, and download data.

**Export Table as** — Use the toolbar above table results to download the current table (CSV/Excel).

**SQL panel** (in the left pane) — The exact SQL query used to fetch the data, along with controls to modify or re-run it.

**Additional tabs** — Visualizations, charts, or summaries appear as separate tabs depending on what the AI generated.

**Pop-out views** — Click the "Open in new pane" icon on any result to view it alongside other results for comparison.

## Save and share

### Export and download

Use **Export Notebook** in the navigation menu to save your current exploration as a Jupyter notebook. This includes all your questions, SQL queries, and visualizations so you can run them again or share them with others.

If you're in Report Mode (accessible via the left sidebar), **Export Notebook** exports all explorations as a single comprehensive notebook.

### Manage your data

Click **Sources** from the left sidebar to:

- **Add Sources** — Upload CSV, Parquet, JSON files or connect to databases
- **View Sources** — See all currently connected data sources and their tables
- Remove data sources you no longer need

All data connections persist within your current session, so you can reference the same datasets across multiple explorations.

### Next steps

Once you're comfortable with the interface, try the quick action buttons to get a feel for what's possible. Then move on to asking more complex questions in the chat. The AI will handle the SQL—you just focus on what you want to learn.
