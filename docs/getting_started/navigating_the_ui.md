# Navigating the Lumen UI

Lumen Explorer combines data exploration and AI-powered analysis in one split-screen interface. This guide shows you where everything is and how to use it.

## Get started

### Your interface at a glance

When you open Lumen Explorer, you'll see the chat panel where you ask questions. The results area (where results appear) and the explorations sidebar (where you manage your work) are collapsed by default and expand as you start analyzing.

**Chat panel** — Ask questions about your data in plain English. The AI will generate queries and visualizations automatically.

**Results area** — View interactive tables, charts, and analysis results. Use Graphic Walker to filter, sort, and explore visually. Appears when you ask your first question.

**Explorations panel** (right sidebar) — Track all your analyses. Each exploration is a separate session, so you can branch off previous results and compare different approaches. Appends as you ask more questions.

### Ask questions

You can ask the AI about your data in two ways:

**Type a direct question** — Enter any question in the "Ask anything..." text box. For example: "What are the top 5 customers by revenue?" or "Show me sales trends over time." Hit enter or click the send arrow.

**Use suggestions** — Below the text box, you'll find four preset buttons that handle common tasks without typing:

- **What data is available?** — Get an instant summary of your dataset's structure and contents.
- **Describe the dataset** — Ask the AI to explain column definitions, data types, and what each field represents.
- **Can you visualize the data?** — Generate an automatic visualization to explore patterns and distributions.
- **Show a demo** — See a pre-built example of what Lumen can do.

### Explore the speed dial

At the left of the chat area, click the **+** icon to reveal a speed dial menu with several actions:

**Clear** — Delete the current message or result.

**Undo** — Revert the last action.

**Rerun** — Re-execute the last query to regenerate results.

**Enhance** — Ask the AI to improve or refine the current result.

**Manage Data** — Add new data sources or replace existing ones. Upload CSV, Parquet, JSON, or connect to external sources.

**Attach files** — Upload files to provide additional context for your questions.

**Select LLM** — Switch between different AI models or configure your LLM provider settings.

## Work with results

### Control how analysis runs

The settings menu in the top-right lets you toggle three analysis features:

**Chain of Thought** — When enabled, the AI shows its reasoning steps. Use this when you want to understand *how* the AI arrived at an answer. Disabled by default.

**SQL Planning** — When enabled, the AI plans its SQL query before executing it. This improves accuracy for complex questions. Enabled by default.

**Validation Step** — When enabled, the AI double-checks results for correctness. This catches data errors early. Enabled by default.

Simply click the toggle next to each option to turn it on or off.

### Organize your explorations

Each time you ask a question, a new "exploration" is created. The left sidebar shows all of them under **Explorations**.

**Home** — Your starting point. This is where all analyses begin. Switch back here anytime.

**Individual explorations** — Each SQL query result gets its own tab labeled with its title (e.g., "Top Customers"). Click any exploration to revisit it.

**Edit mode** — Click the **Edit** toggle to reorder your explorations or delete ones you don't need. When enabled, use the up/down arrows next to each exploration to move it. Click the trash icon to remove it.

### View results

When an analysis completes, you'll see:

**SQL panel** — The exact SQL query used to fetch the data. Scroll this panel if the query is long.

**Overview tab** — An interactive table powered by Graphic Walker. Here you can sort, filter, and download data.

**Additional tabs** — Visualizations, charts, or summaries appear as separate tabs depending on what the AI generated.

## Save and share

### Export and download

At the top of the screen, click **Export Notebook** to save your entire session as a Jupyter notebook. This includes all your questions, SQL queries, and visualizations so you can run them again or share them with others.

If you're in Report Mode (toggled from the header), you can export all explorations as a single comprehensive notebook.

### Manage your data

Click **Manage Data** from the speed dial menu (or the + icon at the left of the chat) to:

- Add new CSV, Parquet, JSON, or other supported data files
- View all currently connected data sources
- Remove data sources you no longer need

All data connections persist within your current session, so you can reference the same datasets across multiple explorations.

### Next steps

Once you're comfortable with the interface, try the quick action buttons to get a feel for what's possible. Then move on to asking more complex questions in the chat. The AI will handle the SQL—you just focus on what you want to learn.
