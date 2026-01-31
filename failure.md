## Failure Analysis: Naive Prompt Stuffing Baseline

### 1. Lost in the Middle (Positional Bias)
The LLM failed to reliably retrieve the injected *“Pranay”* statement when it was placed in the middle of a long document.  
This behavior aligns with the **U-shaped attention distribution** observed in transformer models, where tokens at the beginning and end of a prompt receive disproportionately higher attention than those in the center.

**Impact:**  
Important facts embedded mid-document may be ignored or inconsistently surfaced.

---

### 2. Resource Exhaustion (HTTP 429 – Rate Limiting)
Passing the entire document corpus into a single prompt exceeded the API’s **Tokens Per Minute (TPM)** quota, resulting in `429 Resource Exhausted` errors and blocking further requests.

**Impact:**  
The system is fragile under repeated usage and cannot scale to real-world workloads.

---

### 3. High Latency (Performance Degradation)
Processing an oversized prompt significantly increased **Time to First Token (TTFT)**, making the application feel slow and unresponsive compared to targeted retrieval approaches.

**Impact:**  
Poor user experience for interactive or real-time applications.

---

### 4. Cost Inefficiency
Thousands of irrelevant tokens were processed to extract a single sentence, resulting in an unsustainably high **cost-per-query**.

**Impact:**  
The approach is economically infeasible at scale.

---

### 5. Lack of Traceability (No Citations)
Because all documents were merged into a single prompt string, the model could not provide **document-level or section-level attribution** (e.g., file name, page number, or paragraph).

**Impact:**  
Answers cannot be audited or verified, even when they appear correct.

---

### 6. Context Window Limitations
Even with large-context models, prompt stuffing eventually hits a **hard context window ceiling**, preventing the system from scaling to larger document collections.

**Impact:**  
System scalability is constrained by model context size rather than data size.

---

### 7. Grounding and Hallucination Risk
Without selective retrieval, the model may blend its **parametric knowledge** (training data) with the provided documents instead of treating the documents as the sole source of truth.  
This allows unsupported or spurious statements (e.g., the injected “Pranay” claim) to be accepted and elaborated upon.

**Impact:**  
Increased risk of hallucinated or unverified answers.

---

### Summary
This baseline demonstrates that naive prompt stuffing is **unreliable, expensive, slow, and unsafe**.  
These failures directly motivate the need for **retrieval-based grounding with explicit evidence constraints**, which is addressed in the RAG-based system.
