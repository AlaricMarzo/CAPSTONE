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

// ---------- Local storage helpers ----------
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

export default function ProfilePage({ userEmail, onBack }: ProfilePageProps) {
  const { toast } = useToast();

  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const usernameRef = useRef<HTMLInputElement | null>(null);

  // Seed from email or persisted storage
  const seedProfile = (email?: string): Profile => {
    const persisted = loadPersisted(email);
    if (persisted) return persisted;
    return { username: "admin", fullName: "Admin", email: email ?? "" };
  };

  const [userProfile, setUserProfile] = useState<Profile>(() => seedProfile(userEmail));
  const [draftProfile, setDraftProfile] = useState<Profile>(userProfile);

  useEffect(() => {
    if (isEditing) {
      setDraftProfile(userProfile);
      setTimeout(() => usernameRef.current?.focus(), 0);
    }
  }, [isEditing, userProfile]);

  useEffect(() => {
    const seeded = seedProfile(userEmail);
    setUserProfile(seeded);
    if (!isEditing) setDraftProfile(seeded);
  }, [userEmail, isEditing]);

  const updateDraft = <K extends keyof Profile>(field: K, value: Profile[K]) =>
    setDraftProfile((prev) => ({ ...prev, [field]: value }));

  const handleSaveProfile = async () => {
    if (!isEditing) return;
    setIsSaving(true);
    await new Promise((r) => setTimeout(r, 300));
    const committed: Profile = { ...draftProfile, email: userProfile.email };
    setUserProfile(committed);
    persistProfile(committed);
    toast({ title: "Profile updated", description: "Your profile has been saved." });
    setIsSaving(false);
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setDraftProfile(userProfile);
    setIsEditing(false);
  };

  const isDraftDirty =
    draftProfile.username !== userProfile.username ||
    draftProfile.fullName !== userProfile.fullName;

  const view = isEditing ? draftProfile : userProfile;

  return (
    <div className="flex-1 p-8 pt-6 bg-background">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Back + centered title block (mirrors upload page spacing) */}
        <div className="flex items-center justify-between">
          <Button variant="ghost" size="sm" onClick={onBack}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <div className="flex-1 text-center -ml-8">
            <h1 className="text-4xl font-bold text-foreground">User Profile</h1>
            <p className="mt-2 text-lg text-muted-foreground">
              Manage your account details and preferences
            </p>
          </div>
          <div className="w-20" /> {/* spacer to keep title centered */}
        </div>

        {/* Main card — same visual weight as Upload page cards */}
        <div className="rounded-2xl border border-border bg-card shadow-sm">
          <div className="p-6">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-foreground">Profile Information</h3>
              <p className="text-sm text-muted-foreground">
                View and edit your account details below.
              </p>
            </div>

            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    ref={usernameRef}
                    value={view.username}
                    onChange={(e) => updateDraft("username", e.target.value)}
                    className="shadow-sm"
                    disabled={!isEditing}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={userProfile.email}
                    disabled
                    className="bg-muted shadow-sm"
                  />
                </div>

                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="fullName">Full Name</Label>
                  <Input
                    id="fullName"
                    value={view.fullName}
                    onChange={(e) => updateDraft("fullName", e.target.value)}
                    className="shadow-sm"
                    disabled={!isEditing}
                  />
                </div>
              </div>

              {/* Buttons centered under the form */}
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
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
            </div>
          </div>
        </div>

        {/* Optional spacer card for parity with pages that stack multiple sections */}
        {/* <div className="rounded-2xl border border-border bg-card shadow-sm p-6" /> */}
      </div>
    </div>
  );
}
