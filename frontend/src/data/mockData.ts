export const inventoryData = [
  { id: 1, name: "Paracetamol 500mg", category: "Pain Relief", stock: 5, minStock: 50, price: 12.99, sales: 324 },
  { id: 2, name: "Vitamin D3 1000IU", category: "Vitamins", stock: 0, minStock: 30, price: 24.99, sales: 156 },
  { id: 3, name: "Ibuprofen 400mg", category: "Pain Relief", stock: 87, minStock: 40, price: 15.49, sales: 287 },
  { id: 4, name: "Omega-3 Fish Oil", category: "Supplements", stock: 23, minStock: 25, price: 32.99, sales: 98 },
  { id: 5, name: "Vitamin C 1000mg", category: "Vitamins", stock: 145, minStock: 60, price: 18.99, sales: 234 },
  { id: 6, name: "Aspirin 325mg", category: "Pain Relief", stock: 67, minStock: 45, price: 9.99, sales: 178 },
  { id: 7, name: "Multivitamin Daily", category: "Vitamins", stock: 89, minStock: 35, price: 28.99, sales: 201 },
  { id: 8, name: "Probiotics 50B CFU", category: "Supplements", stock: 12, minStock: 20, price: 45.99, sales: 87 },
];

export const purchaseData = [
  { id: 1, productName: "Paracetamol 500mg", quantity: 2, price: 25.98, customerName: "John Smith", date: "2024-01-15T10:30:00", status: "completed" },
  { id: 2, productName: "Vitamin D3 1000IU", quantity: 1, price: 24.99, customerName: "Sarah Johnson", date: "2024-01-15T14:22:00", status: "completed" },
  { id: 3, productName: "Ibuprofen 400mg", quantity: 3, price: 46.47, customerName: "Mike Brown", date: "2024-01-15T16:45:00", status: "pending" },
  { id: 4, productName: "Omega-3 Fish Oil", quantity: 1, price: 32.99, customerName: "Emily Davis", date: "2024-01-14T09:15:00", status: "completed" },
  { id: 5, productName: "Vitamin C 1000mg", quantity: 2, price: 37.98, customerName: "David Wilson", date: "2024-01-14T11:30:00", status: "completed" },
  { id: 6, productName: "Aspirin 325mg", quantity: 4, price: 39.96, customerName: "Lisa Anderson", date: "2024-01-14T13:20:00", status: "refunded" },
  { id: 7, productName: "Multivitamin Daily", quantity: 1, price: 28.99, customerName: "Robert Taylor", date: "2024-01-13T15:45:00", status: "completed" },
  { id: 8, productName: "Probiotics 50B CFU", quantity: 1, price: 45.99, customerName: "Jennifer Martinez", date: "2024-01-13T17:10:00", status: "completed" },
];

export const salesTrendsData = [
  { date: "Jan 1", sales: 4200, revenue: 12400 },
  { date: "Jan 2", sales: 3800, revenue: 11200 },
  { date: "Jan 3", sales: 5200, revenue: 15600 },
  { date: "Jan 4", sales: 4800, revenue: 14100 },
  { date: "Jan 5", sales: 6200, revenue: 18300 },
  { date: "Jan 6", sales: 5800, revenue: 17200 },
  { date: "Jan 7", sales: 7200, revenue: 21400 },
  { date: "Jan 8", sales: 6800, revenue: 20100 },
  { date: "Jan 9", sales: 5400, revenue: 16200 },
  { date: "Jan 10", sales: 6600, revenue: 19500 },
  { date: "Jan 11", sales: 7800, revenue: 23100 },
  { date: "Jan 12", sales: 8200, revenue: 24600 },
  { date: "Jan 13", sales: 7600, revenue: 22800 },
  { date: "Jan 14", sales: 8800, revenue: 26400 },
  { date: "Jan 15", sales: 9200, revenue: 27600 },
];

export const recentAlerts = [
  {
    id: 1,
    type: 'critical' as const,
    title: 'Out of Stock Alert',
    description: 'Vitamin D3 1000IU is completely out of stock',
    time: '5 minutes ago',
    productId: 2,
  },
  {
    id: 2,
    type: 'warning' as const,
    title: 'Low Stock Warning',
    description: 'Paracetamol 500mg - Only 5 units remaining (Min: 50)',
    time: '15 minutes ago',
    productId: 1,
  },
  {
    id: 3,
    type: 'warning' as const,
    title: 'Low Stock Warning',
    description: 'Probiotics 50B CFU - Only 12 units remaining (Min: 20)',
    time: '1 hour ago',
    productId: 8,
  },
  {
    id: 4,
    type: 'info' as const,
    title: 'Restock Recommended',
    description: 'Omega-3 Fish Oil approaching minimum threshold',
    time: '2 hours ago',
    productId: 4,
  },
  {
    id: 5,
    type: 'success' as const,
    title: 'High Demand Product',
    description: 'Multivitamin Daily sales increased by 45% this week',
    time: '3 hours ago',
    productId: 7,
  },
];

export const productTrafficData = [
  { name: "Paracetamol 500mg", views: 1240, sales: 324, conversionRate: 26.1 },
  { name: "Vitamin C 1000mg", views: 980, sales: 234, conversionRate: 23.9 },
  { name: "Ibuprofen 400mg", views: 1120, sales: 287, conversionRate: 25.6 },
  { name: "Multivitamin Daily", views: 756, sales: 201, conversionRate: 26.6 },
  { name: "Aspirin 325mg", views: 654, sales: 178, conversionRate: 27.2 },
  { name: "Vitamin D3 1000IU", views: 543, sales: 156, conversionRate: 28.7 },
];

export const getUserMetrics = () => ({
  totalRevenue: purchaseData.reduce((sum, purchase) => sum + purchase.price, 0),
  totalSales: purchaseData.filter(p => p.status === 'completed').length,
  averageOrderValue: purchaseData.reduce((sum, purchase) => sum + purchase.price, 0) / purchaseData.length,
  conversionRate: 3.2,
  lowStockItems: inventoryData.filter(item => item.stock <= item.minStock).length,
  outOfStockItems: inventoryData.filter(item => item.stock === 0).length,
  totalProducts: inventoryData.length,
});
