import logging
from random import randint, choice

from locust import task
from locust.contrib.fasthttp import FastHttpUser

# logging
logging.getLogger().setLevel(logging.DEBUG)


class UserBehavior(FastHttpUser):
    @task(1)
    def load(self) -> None:
        """
        Simulates user behaviour.
        :return: None
        """
        logging.debug("Starting user.")
        self.visit_home()
        self.login()
        self.browse()
        # 50/50 chance to buy
        choice_buy = choice([True, False])
        if choice_buy:
            self.buy()
        self.visit_profile()
        self.logout()
        logging.debug("Completed user.")

    def visit_home(self) -> None:
        """
        Visits the landing page.
        :return: None
        """
        # load landing page
        try:
            self.client.get('/')
            logging.debug("Loaded landing page.")
        except Exception as err:
            logging.error(f"Could not load landing page: {err}")

    def login(self) -> None:
        """
        User login with random userid between 1 and 90.
        :return: categories
        """
        # load login page
        try:
            self.client.get('/login')
            logging.debug("Loaded login page.")
        except Exception as err:
            logging.error(f"Could not load login page: {err}")
        # login
        user = randint(1, 99)
        try:
            self.client.post("/loginAction", params={"username": user, "password": "password"})
        except Exception as err:
            logging.error(
                f"Could not login with username: {user} - status: {err}")

    def browse(self) -> None:
        """
        Simulates random browsing behaviour.
        :return: None
        """
        # execute browsing action randomly between 2 and 5 times
        for i in range(1, randint(2, 5)):
            category_id = randint(2, 6)
            page = randint(1, 5)
            try:
                self.client.get("/category", params={"page": page, "category": category_id})
                logging.debug(f"Visited category {category_id} on page 1")
            except Exception as err:
                logging.error(
                    f"Could not visit category {category_id} on page {page}: {err}")
            product_id = randint(7, 506)
            try:
                self.client.get("/product", params={"id": product_id})
                logging.debug(f"Visited product with id {product_id}.")
            except Exception as err:
                logging.error(
                    f"Could not visit product {product_id}: {err}")
            try:
                self.client.post("/cartAction", params={"addToCart": "", "product id": product_id})
                logging.debug(f"Added product {product_id} to cart.")
            except Exception as err:
                logging.error(
                    f"Could not add product {product_id} to cart: status {err}")

    def buy(self) -> None:
        """
        Simulates to buy products in the cart with sample user data.
        :return: None
        """
        # sample user data
        user_data = {
            "firstname": "User",
            "lastname": "User",
            "adress1": "Road",
            "adress2": "City",
            "cardtype": "volvo",
            "cardnumber": "314159265359",
            "expirydate": "12/2050",
            "confirm": "Confirm"
        }
        try:
            self.client.post("/cartAction", params=user_data)
            logging.debug(f"Bought products.")
        except Exception as err:
            logging.error(f"Could not buy products: {err}")

    def visit_profile(self) -> None:
        """
        Visits user profile.
        :return: None
        """
        try:
            self.client.get("/profile")
            logging.debug("Visited profile page.")
        except Exception as err:
            logging.error(f"Could not visit profile page: {err}")

    def logout(self) -> None:
        """
        User logout.
        :return: None
        """
        try:
            self.client.post("/loginAction", params={"logout": ""})
            logging.debug("Successful logout.")
        except Exception as err:
            logging.error(f"Could not log out: {err}")
