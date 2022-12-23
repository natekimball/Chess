use std::fmt::{Display, Formatter, Error};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Player {
    One,
    Two
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Player::One => write!(f, "player 1"),
            Player::Two => write!(f, "player 2"),
        }
    }
}