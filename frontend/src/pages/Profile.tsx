"use client";

import { useEffect, useRef, useState } from "react";
import { Save, ArrowLeft, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface ProfilePageProps {
  userEmail?: string;
  onLogout: () => void;
  onBack?: () => void;
}

type Profile = {
  username: string;
  email: string;
  fullName: string;
};

// ---------- storage helpers ----------
const storageKey = (email: string) => `shield.profile.${email.toLowerCase()}`;

const loadPersisted = (email?: string): Profile | null => {
  if (!email) return null;
  try {
    const raw = localStorage.getItem(storageKey(email));
    return raw ? (JSON.parse(raw) as Profile) : null;
  } catch {
    return null;
  }
};

const persistProfile = (p: Profile) => {
  try {
    localStorage.setItem(storageKey(p.email), JSON.stringify(p));
  } catch {}
};

export default function ProfilePage({ userEmail, onLogout, onBack }: ProfilePageProps) {
  const { toast } = useToast();

  // UI state
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const usernameRef = useRef<HTMLInputElement | null>(null);

  // Seed from email or persisted storage
  const seedProfile = (email?: string): Profile => {
    const persisted = loadPersisted(email);
    if (persisted) return persisted;
    return {
      username: "admin",
      fullName: "Admin",
      email: email ?? "",
    };
  };

  // Persisted profile (truth)
  const [userProfile, setUserProfile] = useState<Profile>(() => seedProfile(userEmail));

  // Draft profile (for edits only)
  const [draftProfile, setDraftProfile] = useState<Profile>(userProfile);

  // When entering edit mode, copy persisted → draft and focus username
  useEffect(() => {
    if (isEditing) {
      setDraftProfile(userProfile);
      setTimeout(() => usernameRef.current?.focus(), 0);
    }
  }, [isEditing]); // eslint-disable-line react-hooks/exhaustive-deps

  // When login email changes (new session), re-seed from storage or defaults
  useEffect(() => {
    const seeded = seedProfile(userEmail);
    setUserProfile(seeded);
    if (!isEditing) setDraftProfile(seeded);
  }, [userEmail]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateDraft = <K extends keyof Profile>(field: K, value: Profile[K]) =>
    setDraftProfile((prev) => ({ ...prev, [field]: value }));

  // Save profile: commit draft -> persisted, write to localStorage, exit edit mode
  const handleSaveProfile = async () => {
    if (!isEditing) return;
    setIsSaving(true);
    await new Promise((r) => setTimeout(r, 300)); // simulate API if needed
    const committed: Profile = { ...draftProfile, email: userProfile.email }; // email fixed to login
    setUserProfile(committed);
    persistProfile(committed);
    toast({ title: "Profile updated", description: "Your profile has been saved." });
    setIsSaving(false);
    setIsEditing(false);
  };

  // Cancel edit: discard draft changes and exit edit mode
  const handleCancelEdit = () => {
    setDraftProfile(userProfile);
    setIsEditing(false);
  };

  // Enable Save only if draft actually changed
  const isDraftDirty =
    draftProfile.username !== userProfile.username ||
    draftProfile.fullName !== userProfile.fullName;

  // Choose which values to show in inputs
  const view = isEditing ? draftProfile : userProfile;

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
                  value={userProfile.email} // always the login email
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
      </div>
    </div>
  );
}
