"use client"

import { useEffect, useRef, useState } from "react"
import { Eye, EyeOff, Save, ArrowLeft, Pencil } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"

interface ProfilePageProps {
  userEmail?: string
  onLogout: () => void
  onBack?: () => void
}

type Profile = {
  username: string
  email: string
  fullName: string
}

export default function ProfilePage({ userEmail, onLogout, onBack }: ProfilePageProps) {
  const { toast } = useToast()

  // ----- Visibility toggles / UI state -----
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isChangingPassword, setIsChangingPassword] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const usernameRef = useRef<HTMLInputElement | null>(null)

  // ----- Seed initial profile from email -----
  const seedFromEmail = (email?: string): Profile => {
    const em = email ?? ""
    const uname = em ? em.split("@")[0] : "capstone"
    const full = uname.replace(/\./g, " ").replace(/\b\w/g, c => c.toUpperCase())
    return { username: uname, email: em, fullName: full }
  }

  // Persisted profile (truth) — this is what the app uses when NOT editing
  const [userProfile, setUserProfile] = useState<Profile>(() => seedFromEmail(userEmail))

  // Draft profile — only used while editing; gets discarded on cancel
  const [draftProfile, setDraftProfile] = useState<Profile>(userProfile)

  // Password form
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  })

  // When entering edit mode, load current persisted profile into draft and focus username
  useEffect(() => {
    if (isEditing) {
      setDraftProfile(userProfile)
      setTimeout(() => usernameRef.current?.focus(), 0)
    }
  }, [isEditing]) // eslint-disable-line react-hooks/exhaustive-deps

  // If the login email prop changes (e.g., a new login), refresh persisted profile
  useEffect(() => {
    if (userEmail) {
      const seeded = seedFromEmail(userEmail)
      setUserProfile(seeded)
      if (!isEditing) setDraftProfile(seeded)
    }
  }, [userEmail]) // eslint-disable-line react-hooks/exhaustive-deps

  // Helpers to update either draft or password states
  const updateDraft = (field: keyof Profile, value: string) =>
    setDraftProfile(prev => ({ ...prev, [field]: value }))
  const updatePassword = (field: string, value: string) =>
    setPasswordForm(prev => ({ ...prev, [field]: value }))

  // Save profile: commit draft -> persisted, exit edit mode
  const handleSaveProfile = async () => {
    if (!isEditing) return
    setIsSaving(true)
    await new Promise(r => setTimeout(r, 600)) // simulate API
    setUserProfile(draftProfile)
    toast({ title: "Profile updated", description: "Your profile has been successfully updated." })
    setIsSaving(false)
    setIsEditing(false)
  }

  // Cancel edit: discard draft changes and exit edit mode
  const handleCancelEdit = () => {
    setDraftProfile(userProfile) // revert any changes
    setIsEditing(false)
  }

  // Detect if draft changed vs persisted to enable/disable Save
  const isDraftDirty =
    draftProfile.username !== userProfile.username ||
    draftProfile.fullName !== userProfile.fullName
  // Email stays read-only here

  // Choose which source to render into inputs
  const view = isEditing ? draftProfile : userProfile

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <h2 className="text-3xl font-bold text-foreground">User Profile</h2>
      </div>

      <div className="grid gap-6 w-full">
        {/* Profile Information Card */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Profile Information</CardTitle>
            <CardDescription>View and edit your account details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  ref={usernameRef}
                  value={view.username}
                  onChange={(e) => updateDraft("username", e.target.value)}
                  className="shadow-soft"
                  disabled={!isEditing}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  value={view.email}
                  disabled
                  className="bg-muted shadow-soft"
                />
              </div>

              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input
                  id="fullName"
                  value={view.fullName}
                  onChange={(e) => updateDraft("fullName", e.target.value)}
                  className="shadow-soft"
                  disabled={!isEditing}
                />
              </div>
            </div>

            {/* Buttons: Edit / Stop Editing + Save Changes */}
            <div className="flex flex-col sm:flex-row gap-2">
              {!isEditing ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsEditing(true)}
                  className="w-full sm:w-auto"
                >
                  <Pencil className="h-4 w-4 mr-2" />
                  Edit
                </Button>
              ) : (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={handleCancelEdit}
                  className="w-full sm:w-auto"
                >
                  <Pencil className="h-4 w-4 mr-2" />
                  Stop Editing
                </Button>
              )}

              <Button
                onClick={handleSaveProfile}
                disabled={!isEditing || !isDraftDirty || isSaving}
                className="w-full sm:w-auto"
              >
                <Save className="h-4 w-4 mr-2" />
                {isSaving ? "Saving..." : "Save Changes"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Password Change Card */}
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle>Security</CardTitle>
            <CardDescription>Manage your password and security settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {!isChangingPassword ? (
              <Button
                variant="outline"
                onClick={() => setIsChangingPassword(true)}
                className="w-full md:w-auto"
              >
                Change Password
              </Button>
            ) : (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="currentPassword">Current Password</Label>
                  <div className="relative">
                    <Input
                      id="currentPassword"
                      type={showCurrentPassword ? "text" : "password"}
                      placeholder="Enter current password"
                      value={passwordForm.currentPassword}
                      onChange={(e) => updatePassword("currentPassword", e.target.value)}
                      className="pr-10 shadow-soft"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    >
                      {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="newPassword">New Password</Label>
                  <div className="relative">
                    <Input
                      id="newPassword"
                      type={showNewPassword ? "text" : "password"}
                      placeholder="Enter new password"
                      value={passwordForm.newPassword}
                      onChange={(e) => updatePassword("newPassword", e.target.value)}
                      className="pr-10 shadow-soft"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                    >
                      {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <div className="relative">
                    <Input
                      id="confirmPassword"
                      type={showConfirmPassword ? "text" : "password"}
                      placeholder="Confirm new password"
                      value={passwordForm.confirmPassword}
                      onChange={(e) => updatePassword("confirmPassword", e.target.value)}
                      className="pr-10 shadow-soft"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    >
                      {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={async () => {
                      setIsSaving(true)
                      if (!passwordForm.currentPassword || !passwordForm.newPassword || !passwordForm.confirmPassword) {
                        toast({ title: "Missing fields", description: "Please fill in all password fields.", variant: "destructive" })
                        setIsSaving(false); return
                      }
                      if (passwordForm.newPassword !== passwordForm.confirmPassword) {
                        toast({ title: "Passwords don't match", description: "New password and confirm password must match.", variant: "destructive" })
                        setIsSaving(false); return
                      }
                      if (passwordForm.newPassword.length < 8) {
                        toast({ title: "Password too short", description: "New password must be at least 8 characters long.", variant: "destructive" })
                        setIsSaving(false); return
                      }
                      await new Promise(r => setTimeout(r, 600))
                      toast({ title: "Password changed", description: "Your password has been successfully updated." })
                      setPasswordForm({ currentPassword: "", newPassword: "", confirmPassword: "" })
                      setIsChangingPassword(false)
                      setIsSaving(false)
                    }}
                    disabled={isSaving}
                  >
                    <Save className="h-4 w-4 mr-2" />
                    {isSaving ? "Updating..." : "Update Password"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setIsChangingPassword(false)
                      setPasswordForm({ currentPassword: "", newPassword: "", confirmPassword: "" })
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
