import { useState } from "react";
import { Sidebar } from "@/components/Layout/Sidebar";
import { MetricCard } from "@/components/Dashboard/MetricCard";
import { RecentAlertsCard } from "@/components/Dashboard/RecentAlertCard";
import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard";
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard";
import {
  TrendingUp,
  DollarSign,
  Package,
  ShoppingCart,
  User,
  Search,
  Calendar,
  FileText
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { getUserMetrics } from "@/data/mockData";
import UploadPage from "./upload";
import ProfilePage from "./Profile";

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }) {
  const metrics = getUserMetrics();
  const [confirmOpen, setConfirmOpen] = useState(false);

  const confirmLogout = () => {
    onLogout();
    setConfirmOpen(false);
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time business insights and analytics</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Last 30 Days</span>
          </div>
          <Button variant="outline" size="sm" className="shadow-soft">
            <FileText className="h-4 w-4 mr-2" />
            Export Report
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onProfileClick}>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive cursor-pointer"
                onSelect={(e) => {
                  e.preventDefault();
                  setConfirmOpen(true);
                }}
              >
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search products, customers, orders..." className="pl-10 shadow-soft" />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Total Revenue" value={`$${metrics.totalRevenue.toFixed(2)}`} change="+12.5% from last month" changeType="positive" icon={DollarSign} color="success" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Total Sales" value={metrics.totalSales.toString()} change="+8.2% from last month" changeType="positive" icon={ShoppingCart} color="info" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Low Stock Items" value={metrics.lowStockItems.toString()} change={`${metrics.outOfStockItems} out of stock`} changeType="warning" icon={Package} color="warning" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Avg Order Value" value={`$${metrics.averageOrderValue.toFixed(2)}`} change="+5.1% from last month" changeType="positive" icon={TrendingUp} color="default" className="shadow-soft hover:shadow-elegant transition-shadow" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2"><SalesTrendsChart /></div>
        <ProductTrafficCard />
      </div>

      <RecentAlertsCard />
    </div>
  );
}

function SalesPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Sales Dashboard</h2>
        <p className="text-muted-foreground">Sales Trends, Forecasting & Product Performance</p>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <p>Sales dashboard content coming soon...</p>
      </div>
    </div>
  );
}

function InventoryPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Inventory</h2>
        <p className="text-muted-foreground">Live Stock Monitoring & Allocation Logic</p>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <p>Inventory management content coming soon...</p>
      </div>
    </div>
  );
}

function ReportsPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Reports</h2>
        <p className="text-muted-foreground">Analytics, Insights & Performance Reports</p>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <p>Reports and analytics coming soon...</p>
      </div>
    </div>
  );
}

export default function Dashboard({ onLogout, userEmail }: { onLogout: () => void; userEmail: string }) {
  const [activeTab, setActiveTab] = useState("home");
  const [previousTab, setPreviousTab] = useState("home");

  const handleProfileClick = () => {
    setPreviousTab(activeTab);
    setActiveTab("profile");
  };

  const handleLogout = () => onLogout();

  const renderContent = () => {
    switch (activeTab) {
      case "home":      return <HomePage onProfileClick={handleProfileClick} onLogout={handleLogout} />;
      case "sales":     return <SalesPage />;
      case "inventory": return <InventoryPage />;
      case "reports":   return <ReportsPage />;
      case "upload":    return <UploadPage />;
      case "profile":   return <ProfilePage userEmail={userEmail} onLogout={handleLogout} onBack={() => setActiveTab(previousTab)} />;
      default:          return <HomePage onProfileClick={handleProfileClick} onLogout={handleLogout} />;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">{renderContent()}</main>
      </div>
    </div>
  );
}
