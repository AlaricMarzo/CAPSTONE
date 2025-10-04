import { useState } from "react";
import { Bell, User, LogOut, Settings, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useToast } from "@/hooks/use-toast";

interface HeaderNavProps {
  username?: string;
}

const notifications = [
  { id: 1, title: "Low Stock Alert", message: "Paracetamol 500mg - Only 5 units left", type: "warning", time: "5m ago" },
  { id: 2, title: "Inventory Critical", message: "Vitamin D3 out of stock", type: "critical", time: "12m ago" },
  { id: 3, title: "Sale Completed", message: "Order #12345 processed successfully", type: "success", time: "1h ago" },
];

export function HeaderNav({ username = "Admin User" }: HeaderNavProps) {
  const [notificationCount, setNotificationCount] = useState(notifications.length);
  const { toast } = useToast();

  const handleLogout = () => {
    toast({
      title: "Logged Out",
      description: "You have been successfully logged out.",
    });
  };

  const handleNotificationClick = () => {
    setNotificationCount(0);
    toast({
      title: "Notifications",
      description: `${notifications.length} notifications viewed`,
    });
  };

  return (
    <div className="flex items-center gap-4">
      {/* Notifications */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="relative" onClick={handleNotificationClick}>
            <Bell className="h-5 w-5" />
            {notificationCount > 0 && (
              <Badge 
                variant="destructive" 
                className="absolute -top-2 -right-2 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs"
              >
                {notificationCount}
              </Badge>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-80">
          <div className="p-3 border-b">
            <h3 className="font-semibold">Notifications</h3>
          </div>
          {notifications.map((notification) => (
            <DropdownMenuItem key={notification.id} className="p-3 cursor-pointer">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${
                    notification.type === 'critical' ? 'bg-destructive' :
                    notification.type === 'warning' ? 'bg-warning' :
                    'bg-success'
                  }`} />
                  <span className="font-medium text-sm">{notification.title}</span>
                  <span className="text-xs text-muted-foreground ml-auto">{notification.time}</span>
                </div>
                <p className="text-sm text-muted-foreground">{notification.message}</p>
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* User Menu */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="gap-2">
            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
              <User className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-medium">{username}</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <div className="p-3 border-b">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center">
                <Shield className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <p className="font-medium">{username}</p>
                <p className="text-sm text-muted-foreground">Administrator</p>
              </div>
            </div>
          </div>
          <DropdownMenuItem className="cursor-pointer">
            <User className="h-4 w-4 mr-2" />
            Profile
          </DropdownMenuItem>
          <DropdownMenuItem className="cursor-pointer">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="cursor-pointer text-destructive" onClick={handleLogout}>
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}