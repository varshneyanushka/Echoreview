const express  = require("express");
const User     = require("../models/User");
const { signToken, protect, restrictTo } = require("../middleware/auth");

const router = express.Router();

// POST /api/auth/login
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: "Email and password required" });

    const user = await User.findOne({ email }).select("+password +active");
    if (!user || !user.active)
      return res.status(401).json({ message: "Invalid credentials" });

    const ok = await user.correctPassword(password, user.password);
    if (!ok)
      return res.status(401).json({ message: "Invalid credentials" });

    const token = signToken(user._id);
    res.json({
      token,
      user: { _id: user._id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Server error" });
  }
});

// POST /api/auth/register  (admin only)
router.post("/register", protect, restrictTo("admin"), async (req, res) => {
  try {
    const { name, email, password, role } = req.body;
    const user = await User.create({ name, email, password, role });
    const token = signToken(user._id);
    res.status(201).json({
      token,
      user: { _id: user._id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    if (err.code === 11000)
      return res.status(400).json({ message: "Email already registered" });
    res.status(500).json({ message: "Server error" });
  }
});

// GET /api/auth/me
router.get("/me", protect, (req, res) => {
  const { _id, name, email, role } = req.user;
  res.json({ user: { _id, name, email, role } });
});

module.exports = router;