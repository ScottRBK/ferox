# About
ferox is an llm provider gateway

This project is currently being implemented as an upskilling exercise

## Guidelines
You operate as pair programmer primarily

**CRITICAL** Do not modify any files unless you are explicity instructed to do so.

Ideally please refer to patterns and methods described in 
[The Rust Programming](https://doc.rust-lang.org/book/title-page.html)
or [Microsoft/RustTraining](https://github.com/microsoft/RustTraining/tree/main/python-book) repo.

## [Architecture](./docs/design/architecture.md)
Please read to understand architectural preference, when you need to understand the code or, in the 
rate cases where you are asked to do so, make changes to the code.

### Test Approach

1. Unit Tests - module tests held within a related file/module they are validating behaviour on
1. Integration Tests - held within the `tests/integration` folder, test the seams between modules without
invoking actual interface boundaries
1. e2e Tests - held within the `tests/e2e` folder,  tests that test a full e2e user journey, invoke
the interface boundaries, eg/ make actual model calls, these are feature switched in `./Cargo.toml`

```bash
cargo test # runs unit and integartion tests 
```

```bash
cargo test --features e2e-tests --test e2e # only runs the E2E group
```

```bash
cargo test --features e2e-tests # tests everything including the e2e
```

