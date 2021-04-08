import logging
from random import choice
from random import randint

from locust import HttpUser, task, constant


class UserBehavior(HttpUser):
    wait_time = constant(1)

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        print('Starting')
        self.login()

    def login(self):
        credentials = {
            'name': 'user',
            'password': 'password'
        }
        res = self.client.post('/api/user/login', json=credentials)
        logging.info('login {}'.format(res.status_code))

    @task
    def load(self):
        uniqueid = 1
        self.client.get('/')

        uniqueid_request = self.client.get('/api/user/uniqueid')
        if uniqueid_request.status_code == 200:
            user = uniqueid_request.json()
            uniqueid = user['uuid']
            logging.info('User {}'.format(uniqueid))
        else:
            logging.error(f"Could not get unique id.")

        self.client.get('/api/catalogue/categories')
        # all products in catalogue
        products_request = self.client.get('/api/catalogue/products')
        if products_request.status_code == 200:
            products = products_request.json()
            for i in range(2):
                item = None
                while True:
                    item = choice(products)
                    if item['instock'] != 0:
                        break
                # vote for item
                self.client.put('/api/ratings/api/rate/{}/{}'.format(item['sku'], randint(1, 5)))
                self.client.get('/api/catalogue/product/{}'.format(item['sku']))
                self.client.get('/api/ratings/api/fetch/{}'.format(item['sku']))
                self.client.get('/api/cart/add/{}/{}/1'.format(uniqueid, item['sku']))
        else:
            logging.error(f"Could not get products with ip: {uniqueid}")

        cart_request = self.client.get('/api/cart/cart/{}'.format(uniqueid))
        if cart_request.status_code == 200:
            cart = cart_request.json()
            item = choice(cart['items'])
            self.client.get('/api/cart/update/{}/{}/2'.format(uniqueid, item['sku']))
        else:
            logging.error(f"Could not get cart with ip: {uniqueid}")

        # country codes
        code_request = self.client.get('/api/shipping/codes')
        if code_request.status_code == 200:
            code = choice(code_request.json())
            city_request = self.client.get('/api/shipping/cities/{}'.format(code['code']))
            if city_request.status_code == 200:
                city = choice(city_request.json())
                logging.info('code {} city {}'.format(code, city))
                shipping_request = self.client.get('/api/shipping/calc/{}'.format(city['uuid']))
                if shipping_request.status_code == 200:
                    shipping = shipping_request.json()
                    shipping['location'] = '{} {}'.format(code['name'], city['name'])
                    logging.info('Shipping {}'.format(shipping))
                    # POST
                    cart_request = self.client.post('/api/shipping/confirm/{}'.format(uniqueid), json=shipping)
                    if cart_request.status_code == 200:
                        cart = cart_request.json()
                        logging.info('Final cart {}'.format(cart))
                        order_request = self.client.post('/api/payment/pay/{}'.format(uniqueid), json=cart)
                        if order_request.status_code == 200:
                            order = order_request.json()
                            logging.info('Order {}'.format(order))
                        else:
                            logging.error(f"Could not post order with ip: {uniqueid}")
                            logging.error(code_request)
                    else:
                        logging.error(f"Could not post cart with ip: {uniqueid}")
                        logging.error(cart_request)
                else:
                    logging.error(f"Could not get shipping with ip: {uniqueid}")
                    logging.error(shipping_request)
            else:
                logging.error(f"Could not get city with ip: {uniqueid}")
                logging.error(city_request)
        else:
            logging.error(f"Could not get code with ip: {uniqueid}")
            logging.error(code_request)
