const jwt  = require("jsonwebtoken");
const User = require("../models/User");

const JWT_SECRET  = process.env.JWT_SECRET  || "echoreview_dev_secret_change_in_prod";
const JWT_EXPIRES = process.env.JWT_EXPIRES || "7d";

// ── Sign a token ──────────────────────────────────────────────────────────────
function signToken(userId) {
  return jwt.sign({ id: userId }, JWT_SECRET, { expiresIn: JWT_EXPIRES });
}

// ── Protect middleware (requires Bearer token) ────────────────────────────────
async function protect(req, res, next) {
  try {
    let token;

    // Accept from Authorization header OR ?token= query param (needed for SSE / EventSource)
    if (req.headers.authorization?.startsWith("Bearer ")) {
      token = req.headers.authorization.split(" ")[1];
    } else if (req.query.token) {
      token = req.query.token;
    }

    if (!token) {
      return res.status(401).json({ message: "Not authenticated — no token provided" });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    const user    = await User.findById(decoded.id).select("+active");

    if (!user || !user.active) {
      return res.status(401).json({ message: "User no longer active" });
    }

    req.user = user;
    next();
  } catch (err) {
    return res.status(401).json({ message: "Invalid or expired token" });
  }
}

// ── Role guard ────────────────────────────────────────────────────────────────
function restrictTo(...roles) {
  return (req, res, next) => {
    if (!roles.includes(req.user?.role)) {
      return res.status(403).json({ message: "You do not have permission for this action" });
    }
    next();
  };
}

module.exports = { signToken, protect, restrictTo };