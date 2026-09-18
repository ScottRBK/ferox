# About
ferox is an llm provider gateway

This project is currently being implemented as an upskilling exercise

## Guidelines
You operate as pair programmer primarily

**CRITICAL** Do not modify any files unless you are explicity instructed to do so.

## [Architecture](./docs/design/architecture.md)
Please read to understand architectural preference, when you need to understand the code or, in the 
rate cases where you are asked to do so, make changes to the code.

### Test Approach

1. Unit Tests - module tests held within a related file/module they are validating behaviour on
1. Integration Tests - held within the `tests/integration` folder, test the seams between modules without
invoking actual interface boundaries
1. e2e Tests - held within the `tests/e2e` folder,  tests that test a full e2e user journey, invoke
the interface boundaries, eg/ make actual model calls, these are feature switched in `./Cargo.toml`

#### runs unit and integartion tests 
```bash
cargo test 
```

#### only runs the E2E group
```bash
cargo test --features e2e-tests --test e2e 
```

#### tests everything including the e2e
```bash
cargo test --features e2e-tests 
```

