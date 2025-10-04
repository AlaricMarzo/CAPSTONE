import { cn } from "@/lib/utils";
import { 
  Home, 
  ShoppingCart, 
  Package, 
  FileBarChart, 
  Users, 
  Settings,
  Bell
} from "lucide-react";

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const navigation = [
  { id: 'home', name: 'Home', icon: Home },
  { id: 'sales', name: 'Sales', icon: ShoppingCart },
  { id: 'inventory', name: 'Inventory', icon: Package },
  { id: 'reports', name: 'Reports', icon: FileBarChart },
  { id: 'users', name: 'Users', icon: Users },
  { id: 'settings', name: 'Settings', icon: Settings },
];

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  return (
    <div className="flex h-screen w-64 flex-col bg-sidebar-background border-r border-sidebar-border">
      {/* Logo */}
      <div className="flex h-16 items-center border-b border-sidebar-border px-6">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">S</span>
          </div>
          <span className="text-xl font-bold text-sidebar-foreground">SHIELD</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-4 py-4">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-2 py-2">
          Payments
        </div>
        {navigation.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                activeTab === item.id
                  ? "bg-primary text-primary-foreground"
                  : "text-sidebar-foreground hover:bg-secondary hover:text-secondary-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.name}
            </button>
          );
        })}
      </nav>

      {/* User Section */}
      <div className="border-t border-sidebar-border p-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center">
            <span className="text-sm font-medium">U</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-sidebar-foreground">Username</p>
            <p className="text-xs text-muted-foreground">ID: 1543907</p>
          </div>
          <Bell className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
    </div>
  );
}