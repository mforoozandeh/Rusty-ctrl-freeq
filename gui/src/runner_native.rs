//! Native runner: the optimisation on a background thread, messages down a channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use ctrl_freeq::config::Config;

use crate::run::{MessageSink, RunMessage, run_to_sink};

struct ChannelSink {
    tx: Sender<RunMessage>,
    cancel: Arc<AtomicBool>,
}

impl MessageSink for ChannelSink {
    fn send(&mut self, message: RunMessage) {
        // A closed channel means the interface has gone; the cancel flag stops the run at its next check.
        let _ = self.tx.send(message);
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A run in progress.
pub struct Runner {
    rx: Receiver<RunMessage>,
    cancel: Arc<AtomicBool>,
}

impl Runner {
    /// Start `config` on a background thread.
    pub fn start(config: Config) -> Self {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut sink = ChannelSink {
            tx,
            cancel: Arc::clone(&cancel),
        };
        std::thread::spawn(move || run_to_sink(&config, &mut sink));
        Runner { rx, cancel }
    }

    /// Every message that has arrived since the last call.
    pub fn drain(&mut self) -> Vec<RunMessage> {
        self.rx.try_iter().collect()
    }

    /// Ask the optimiser to stop.  It finishes its current step and the best point so far is analysed and
    /// returned as usual.
    pub fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether a cancelled run still delivers its best pulse on this platform.
    pub const CANCEL_KEEPS_RESULT: bool = true;
}
