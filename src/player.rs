use std::fmt::{Display, Formatter, Error};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Player {
    One,
    Two
}

impl Player {
    pub fn number(&self) -> u8 {
        match self {
            Player::One => 1,
            Player::Two => 2,
        }
    }

    pub(crate) fn other(&self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }

    pub(crate) fn is_maximizing(&self) -> bool {
        matches!(self, Player::One)
    }
}

impl Display for Player {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Player::One => write!(f, "player 1"),
            Player::Two => write!(f, "player 2"),
        }
    }
}