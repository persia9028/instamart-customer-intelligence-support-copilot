-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2 Revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o JOIN customers c USING (customer_id) JOIN products p USING (product_id)
WHERE o.status = 'completed'
GROUP BY c.city, p.category ORDER BY c.city, p.category;

-- Q3 Products priced above catalogue average
SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4 Customers without a completed order
SELECT c.customer_id FROM customers c LEFT JOIN orders o
ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5 Customers whose spend is above the average customer spend
WITH customer_spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status = 'completed' GROUP BY customer_id
), average_spend AS (SELECT AVG(total_spend) AS avg_spend FROM customer_spend)
SELECT customer_spend.customer_id, customer_spend.total_spend
FROM customer_spend CROSS JOIN average_spend
WHERE customer_spend.total_spend > average_spend.avg_spend ORDER BY customer_spend.customer_id;

-- Q6 Customer spend ranking within city
WITH spend AS (
  SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM customers c JOIN orders o USING (customer_id)
  WHERE o.status = 'completed' GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS city_row
FROM spend ORDER BY city, spend_rank, customer_id;

-- Q7 Daily revenue and cumulative revenue
WITH daily AS (
  SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
  FROM orders WHERE status = 'completed' GROUP BY substr(order_ts, 1, 10)
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily ORDER BY day;
