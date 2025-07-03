```
┌─────────────────────────────────────────────────────────────┐
│ HIGH-LEVEL ROADMAP (Generated Once)                        │
│ ┌─ [ ] Find database schemas                               │
│ ├─ [ ] Find customers with most orders                    │
│ └─ [ ] Generate analysis report                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ MILESTONE EXECUTION (Decisive Loop)                        │
│ Current: "Find database schemas"                           │
│ ┌─ Iteration 1: execute_queries → ✅ Found tables         │
│ ├─ Iteration 2: execute_queries → ✅ Found relationships  │
│ └─ Auto-complete: Sufficient progress achieved            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ ATOMIC ACTIONS (Existing @action methods)                  │
│ @action execute_queries(instruction)                       │
│ @action generate_query(instruction)                        │
│ @action edit_lines(query, feedback)                        │
└─────────────────────────────────────────────────────────────┘
```
