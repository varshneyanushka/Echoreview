const mongoose = require("mongoose");
const bcrypt   = require("bcryptjs");

const UserSchema = new mongoose.Schema(
  {
    name:     { type: String, required: true, trim: true },
    email:    { type: String, required: true, unique: true, lowercase: true, trim: true },
    password: { type: String, required: true, select: false },
    role:     { type: String, enum: ["admin", "agent"], default: "agent" },
    active:   { type: Boolean, default: true, select: false },
  },
  { timestamps: true }
);

// Hash password on save
UserSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

UserSchema.methods.correctPassword = function (candidate, stored) {
  return bcrypt.compare(candidate, stored);
};

module.exports = mongoose.model("User", UserSchema);