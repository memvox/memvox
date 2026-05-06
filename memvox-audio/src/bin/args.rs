use cpal::traits::{DeviceTrait, HostTrait};
use std::env;

#[derive(Clone, Default)]
struct Args {
    name: String,
    city: String,
    car: Option<String>,
    married: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();

    list_devices();

    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };

    Args {
        name: get("--name").unwrap_or_else(|| "Barney".to_string()),
        city: get("--city").unwrap_or_else(|| "Seattle".to_string()),
        car: get("--car"),
        married: argv.iter().any(|a| a == "--married"),
    }
}

fn list_devices() {
    let host = cpal::default_host();
    println!("cpal host: {}", host.id().name());

    match host.input_devices() {
        Ok(devs) => {
            println!("  inputs:");
            for d in devs {
                let name = d.name().unwrap_or_else(|_| "<unnamed>".into());
                let cfg = d
                    .default_input_config()
                    .map(|c| {
                        format!(
                            "{} Hz, {} ch, {:?}",
                            c.sample_rate().0,
                            c.channels(),
                            c.sample_format()
                        )
                    })
                    .unwrap_or_else(|_| "no default config".into());
                println!("    * {}   ({})", name, cfg);
            }
        }
        Err(e) => println!("  inputs: error: {}", e),
    }

    match host.output_devices() {
        Ok(devs) => {
            println!("  outputs:");
            for d in devs {
                let name = d.name().unwrap_or_else(|_| "<unnamed>".into());
                let cfg = d
                    .default_output_config()
                    .map(|c| {
                        format!(
                            "{} Hz, {} ch, {:?}",
                            c.sample_rate().0,
                            c.channels(),
                            c.sample_format()
                        )
                    })
                    .unwrap_or_else(|_| "no default config".into());
                println!("    * {}   ({})", name, cfg);
            }
        }
        Err(e) => println!("  outputs: error: {}", e),
    }
    println!("---------------------------------------------------");
}

fn main() {
    let args = parse_args();
    println!("name: {}", args.name);
    println!("city: {}", args.city);
    println!("car: {}", args.car.unwrap_or("Mercedes".to_string()));
    println!("married: {}", args.married);
}
