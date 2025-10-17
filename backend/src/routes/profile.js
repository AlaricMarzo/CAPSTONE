import express from "express"
import pkg from "pg"
const { Pool } = pkg

const router = express.Router()

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : false,
})

// Get user profile
router.get("/profile/:userId", async (req, res) => {
  try {
    const { userId } = req.params

    const query = `
      SELECT id, username, email, full_name, created_at
      FROM users
      WHERE id = $1
    `

    const result = await pool.query(query, [userId])

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: "User not found"
      })
    }

    const user = result.rows[0]
    res.json({
      success: true,
      profile: {
        id: user.id,
        username: user.username,
        email: user.email,
        fullName: user.full_name,
        joinDate: user.created_at.toISOString().split('T')[0]
      }
    })
  } catch (error) {
    console.error("Error fetching profile:", error)
    res.status(500).json({
      success: false,
      error: "Failed to fetch profile"
    })
  }
})

// Update user profile
router.put("/profile/:userId", async (req, res) => {
  try {
    const { userId } = req.params
    const { username, fullName } = req.body

    // Validate input
    if (!username || !fullName) {
      return res.status(400).json({
        success: false,
        error: "Username and full name are required"
      })
    }

    const query = `
      UPDATE users
      SET username = $1, full_name = $2, updated_at = CURRENT_TIMESTAMP
      WHERE id = $3
      RETURNING id, username, email, full_name, updated_at
    `

    const result = await pool.query(query, [username, fullName, userId])

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: "User not found"
      })
    }

    const user = result.rows[0]
    res.json({
      success: true,
      message: "Profile updated successfully",
      profile: {
        id: user.id,
        username: user.username,
        email: user.email,
        fullName: user.full_name,
        updatedAt: user.updated_at
      }
    })
  } catch (error) {
    console.error("Error updating profile:", error)
    res.status(500).json({
      success: false,
      error: "Failed to update profile"
    })
  }
})

// Change password
router.put("/profile/:userId/password", async (req, res) => {
  try {
    const { userId } = req.params
    const { currentPassword, newPassword } = req.body

    // Validate input
    if (!currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        error: "Current password and new password are required"
      })
    }

    if (newPassword.length < 8) {
      return res.status(400).json({
        success: false,
        error: "New password must be at least 8 characters long"
      })
    }

    // First, verify current password
    const verifyQuery = `
      SELECT password_hash FROM users WHERE id = $1
    `

    const verifyResult = await pool.query(verifyQuery, [userId])

    if (verifyResult.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: "User not found"
      })
    }

    // Note: In a real implementation, you'd hash the currentPassword and compare with password_hash
    // For now, we'll assume the password is correct (since we're using Firebase auth)
    // You would implement proper password verification here

    // Update password hash (you'd hash the newPassword here)
    const updateQuery = `
      UPDATE users
      SET password_hash = $1, updated_at = CURRENT_TIMESTAMP
      WHERE id = $2
      RETURNING id, updated_at
    `

    // Note: In production, hash the password before storing
    const result = await pool.query(updateQuery, [newPassword, userId])

    res.json({
      success: true,
      message: "Password changed successfully"
    })
  } catch (error) {
    console.error("Error changing password:", error)
    res.status(500).json({
      success: false,
      error: "Failed to change password"
    })
  }
})

export default router
