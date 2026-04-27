// use settings::Settings;

mod settings;
mod cli; 

#[tokio::main]
async fn main() {

    // let settings = Settings::new();
    cli::repl().await.expect("repl failed");
}
