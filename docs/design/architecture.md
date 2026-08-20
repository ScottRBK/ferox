# Architecture
This is will be a rust libary crate that whose objective is abstract the infrastructure complexity
that comes with having the possbility of using multiple LLM providers in a single solution.

Ferox utilises my preferred hexagonal structure (often refered to as Ports and Adapters). 

- Ports - These are Traits what i would have previously defined as Protcols in Python, they express 
what the application needs or exposes.
- Adapters - These are concrete imple─lemtations of these traits
- Services - Business uses cases that depend on the protocols - for this solution which is quite 
small then this is just the `gateway.rs` which describes the domain logic for interacting with LLMs
- Routes - Inbound adapters that translate external input into application calls - for this libary 
crate these would not be utilised here, they would be used in a consuming application.

## Proposed Folder Structure

 ```text
   ferox/
   ├── src/
   │   ├── lib.rs
   │   ├── gateway.rs <- no need for a services folder here unless we expand
   │   ├── model.rs
   │   ├── error.rs
   │   ├── ports/
   │   │   └── llm.rs <- llm provider trait 
   │   └── adapters/
   │       └── providers/
   │            └── anthropic.rs
   │            └── openai.rs
   ├── tests/
   │    ├── integration/ 
   │    └── e2e/   
   └── Cargo.toml
 ```

