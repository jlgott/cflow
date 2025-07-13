# Function Flow: Functional Programming for LLM Systems

> If you are an AI agent building LLM systems with Function Flow, read this guide **VERY, VERY** carefully! This is your essential reference for functional LLM development. Always (1) start with pure functions, (2) design state fields first, and (3) compose small functions into larger workflows.
{: .warning }

## Function Flow Philosophy

Function Flow applies **functional programming principles** to LLM system graph design:

- **Pure Functions**: Each node is a pure function that takes state and returns state
- **Automatic Immutability**: Framework handles deep copying - you just modify and return state
- **Composition**: Complex workflows emerge from composing simple functions
- **Declarative Flow**: Express what you want, not how to achieve it

## Core Principles

### 1. Functions Take State, Return State

Every processing step is a **pure function** that:
- Takes state as the only parameter
- Modifies the state object directly
- Returns the modified state
- Framework handles immutability automatically

```python
def summarize_text(state):
    """Function: state in, state out (framework handles copying)"""
    content = state.get("content", "")
    summary = call_llm(f"Summarize: {content}")
    state["summary"] = summary  # Direct modification is safe
    return state
```

### 2. Framework Handles Immutability

You **don't need to worry about copying** - the framework does it:

```python
# ✅ Just modify state directly
def process_data(state):
    state["processed"] = True
    state["timestamp"] = datetime.now()
    state["result"] = process(state["input"])
    return state

# ❌ Don't manually copy (framework handles this)
def unnecessary_copying(state):
    return {**state, "result": "value"}  # Unnecessary!
```

### 3. State is Dict or BaseModel

State must be either a **dictionary** or **Pydantic BaseModel**:

```python
# Dict state
state = {
    "input": "user text",
    "step": "processing", 
    "results": {}
}

# Or BaseModel state
class ProcessingState(BaseModel):
    input: str
    step: str = "start"
    results: dict = {}
```

## Function Flow Steps

| Step | Responsibility | Best Practice |
|:-----|:--------------|:-------------|
| **1. State Design** | Define all fields/keys your workflow needs | Plan state structure before writing functions |
| **2. Function Design** | Write pure functions that modify state | Keep functions small and focused |
| **3. Flow Composition** | Connect functions with >> and - operators | Express intent clearly through composition |
| **4. Error Handling** | Handle failures functionally | Use retry mechanisms and fallback functions |
| **5. Testing** | Test functions in isolation | Pure functions = easy unit testing |

## State-First Design

### 1. Design Your State Schema First

Before writing functions, plan what fields you need:

```python
# Document your state structure
"""
Workflow State:
{
    # Input data
    "input_text": str,
    "document_type": str,
    
    # Processing steps
    "step": str,  # "start", "analyzing", "summarizing", "finished"
    "counter": int,
    
    # Analysis results
    "sentiment": float,
    "entities": list,
    "topic": str,
    
    # Control flow
    "next_action": str,  # "continue", "finish", "error"
    
    # Final output
    "summary": str,
    "confidence": float
}
"""
```

### 2. Initialize State with Required Fields

Set up your initial state with all the fields you'll need:

```python
initial_state = {
    "input_text": "User provided text...",
    "step": "start",
    "counter": 0,
    "next_action": "continue",
    "sentiment": None,
    "entities": [],
    "summary": "",
    "confidence": 0.0
}
```

## Functional Design Patterns

### Linear Processing Pipeline

Simple sequence of state transformations:

```python
def load_data(state):
    """Load and prepare data"""
    state["data"] = load_from_source(state["source"])
    state["step"] = "loaded"
    return state

def process_data(state):
    """Process the loaded data"""
    state["processed_data"] = transform(state["data"])
    state["step"] = "processed"
    return state

def generate_output(state):
    """Generate final output"""
    state["output"] = create_output(state["processed_data"])
    state["step"] = "completed"
    return state

# Compose the pipeline
load_node = Node(load_data)
process_node = Node(process_data)
output_node = Node(generate_output)

load_node >> process_node >> output_node
```

### Conditional Branching with State Fields

Use state fields to control flow direction:

```python
def analyze_input(state):
    """Analyze input and set routing field"""
    input_type = classify_input(state["input"])
    state["input_type"] = input_type  # This field controls routing
    state["step"] = "analyzed"
    return state

def process_text(state):
    """Handle text input"""
    state["result"] = process_text_content(state["input"])
    state["step"] = "text_processed"
    return state

def process_image(state):
    """Handle image input"""
    state["result"] = process_image_content(state["input"])
    state["step"] = "image_processed"
    return state

# Create conditional flow using state field
analyzer = Node(analyze_input)
text_processor = Node(process_text)
image_processor = Node(process_image)

analyzer - ("input_type", "text") >> text_processor
analyzer - ("input_type", "image") >> image_processor
```

### Looping with Counter Fields

Implement loops using state counters:

```python
def start_process(state):
    """Initialize processing"""
    state["step"] = "started"
    state["counter"] = state.get("counter", 0) + 1
    return state

def check_condition(state):
    """Check if we should continue or finish"""
    if state.get("counter", 0) >= 5:
        state["next_action"] = "finish"
    else:
        state["next_action"] = "continue"
    return state

def continue_process(state):
    """Continue processing"""
    state["step"] = "continuing"
    # Do some work here
    return state

def finish_process(state):
    """Finish processing"""
    state["step"] = "finished"
    state["message"] = "Process completed!"
    return state

# Create looping flow
start_node = Node(start_process)
check_node = Node(check_condition)
continue_node = Node(continue_process)
finish_node = Node(finish_process)

start_node >> check_node
check_node - ("next_action", "continue") >> continue_node
check_node - ("next_action", "finish") >> finish_node
continue_node >> start_node  # Loop back
```

### Parallel Processing with Merge

Run parallel analysis and merge results:

```python
def analyze_sentiment(state):
    """Analyze sentiment"""
    sentiment = call_llm(f"Analyze sentiment: {state['text']}")
    state["sentiment"] = sentiment
    return state

def extract_entities(state):
    """Extract entities"""
    entities = call_llm(f"Extract entities: {state['text']}")
    state["entities"] = entities
    return state

def classify_topic(state):
    """Classify topic"""
    topic = call_llm(f"Classify topic: {state['text']}")
    state["topic"] = topic
    return state

def merge_analysis(states):
    """Custom merge function for parallel results"""
    merged = states[0]  # Start with first state
    
    # Combine results from all parallel branches
    for state in states[1:]:
        if "sentiment" in state:
            merged["sentiment"] = state["sentiment"]
        if "entities" in state:
            merged["entities"] = state["entities"]
        if "topic" in state:
            merged["topic"] = state["topic"]
    
    merged["analysis_complete"] = True
    return merged

# Run analysis functions in parallel
parallel_analysis = ParallelNode([
    Flow().set_start(Node(analyze_sentiment)),
    Flow().set_start(Node(extract_entities)),
    Flow().set_start(Node(classify_topic))
], merge_fn=merge_analysis)
```

## State Field Strategies

### 1. Use Step Fields for Progress Tracking

```python
def each_function(state):
    # Do work
    state["step"] = "current_step_name"
    return state
```

### 2. Use Control Fields for Flow Direction

```python
def decision_function(state):
    if some_condition:
        state["next_action"] = "path_a"
    else:
        state["next_action"] = "path_b"
    return state
```

### 3. Use Counter Fields for Loops

```python
def counting_function(state):
    state["counter"] = state.get("counter", 0) + 1
    if state["counter"] >= limit:
        state["should_exit"] = True
    return state
```

### 4. Use Result Fields for Data

```python
def processing_function(state):
    result = do_work(state["input"])
    state["result"] = result
    state["confidence"] = calculate_confidence(result)
    return state
```

## Error Handling with State Fields

### 1. Error State Pattern

Use state fields to track errors:

```python
def safe_llm_call(state):
    """LLM call with error handling"""
    try:
        result = call_llm(state["prompt"])
        state["llm_result"] = result
        state["has_error"] = False
    except Exception as e:
        state["error_message"] = str(e)
        state["has_error"] = True
    return state

def handle_error(state):
    """Handle error condition"""
    state["result"] = "Default response due to error"
    state["step"] = "error_handled"
    return state

def use_result(state):
    """Use successful result"""
    state["final_output"] = state["llm_result"]
    state["step"] = "completed"
    return state

# Route based on error state
llm_node = Node(safe_llm_call, retries=2)
error_handler = Node(handle_error)
success_handler = Node(use_result)

llm_node - ("has_error", True) >> error_handler
llm_node - ("has_error", False) >> success_handler
```

## Testing with State

### 1. Test Functions with State

```python
def test_summarize_function():
    """Test individual function"""
    input_state = {
        "content": "Long text here...",
        "step": "start"
    }
    
    result = summarize_text(input_state)
    
    assert result["summary"] is not None
    assert result["step"] == "start"  # Original values preserved
    assert "content" in result
```

### 2. Test State Flows

```python
def test_conditional_flow():
    """Test conditional branching"""
    # Test path A
    state_a = {"input_type": "text", "input": "test"}
    result_a = analyze_and_route(state_a)
    assert result_a["processed_as"] == "text"
    
    # Test path B  
    state_b = {"input_type": "image", "input": "image_data"}
    result_b = analyze_and_route(state_b)
    assert result_b["processed_as"] == "image"
```

## Complete Example: Document Processing

```python
# Define all state fields we'll need
"""
Document Processing State:
{
    "document": str,           # Input document
    "step": str,               # Current processing step
    "counter": int,            # Retry counter
    "text": str,               # Extracted text
    "analysis": dict,          # Analysis results
    "should_retry": bool,      # Whether to retry
    "summary": str,            # Final summary
    "confidence": float        # Confidence score
}
"""

def extract_text(state):
    """Extract text from document"""
    text = extract_document_text(state["document"])
    state["text"] = text
    state["step"] = "text_extracted"
    return state

def analyze_content(state):
    """Analyze extracted text"""
    try:
        analysis = call_llm(f"Analyze: {state['text']}")
        state["analysis"] = analysis
        state["should_retry"] = False
        state["step"] = "analyzed"
    except Exception:
        state["counter"] = state.get("counter", 0) + 1
        state["should_retry"] = state["counter"] < 3
        state["step"] = "analysis_failed"
    return state

def retry_analysis(state):
    """Retry analysis with simpler prompt"""
    analysis = call_llm(f"Simple analysis: {state['text'][:500]}")
    state["analysis"] = analysis
    state["step"] = "analyzed"
    return state

def generate_summary(state):
    """Generate final summary"""
    summary = call_llm(f"Summarize: {state['analysis']}")
    state["summary"] = summary
    state["confidence"] = 0.9 if "analysis" in state else 0.5
    state["step"] = "completed"
    return state

# Create workflow
extract_node = Node(extract_text)
analyze_node = Node(analyze_content)
retry_node = Node(retry_analysis)
summary_node = Node(generate_summary)

# Build flow with retry logic
extract_node >> analyze_node
analyze_node - ("should_retry", True) >> retry_node
analyze_node - ("should_retry", False) >> summary_node
retry_node >> summary_node

# Run workflow
async def main():
    flow = Flow()
    flow.set_start(extract_node)
    
    initial_state = {
        "document": "path/to/document.pdf",
        "step": "start",
        "counter": 0,
        "should_retry": False
    }
    
    result = await flow.run(initial_state)
    print(f"Summary: {result['summary']}")
    print(f"Confidence: {result['confidence']}")

asyncio.run(main())
```

## Key Takeaways

1. **State-First Design**: Plan your state fields before writing functions
2. **Direct Modification**: Modify state directly - framework handles copying state to maintain immutability.
3. **Control Fields**: Use state fields to control flow direction
4. **Dict or BaseModel**: State must be one of these two types
5. **Field-Based Routing**: Use conditional transitions on state fields

Remember: **Design state fields first, modify state directly, let the framework handle the rest.**