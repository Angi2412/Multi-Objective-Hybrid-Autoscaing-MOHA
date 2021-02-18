import os

from locust import HttpUser, task, between
from random import choice
from random import randint
from faker import Factory
from json.decoder import JSONDecodeError
import logging


class UserBehavior(HttpUser):
    wait_time = between(0.5, 5)
    fake = Factory.create()
    fake_ip = fake.ipv4(network=False)

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        print('Starting')

    @task
    def login(self):
        credentials = {
            'name': 'user',
            'password': 'password'
        }
        res = self.client.post('/api/user/login', json=credentials, headers={'x-forwarded-for': self.fake_ip})
        print('login {}'.format(res.status_code))

    @task
    def load(self):
        res = self.client.get('/', headers={'x-forwarded-for': self.fake_ip})
        user = self.client.get('/api/user/uniqueid', headers={'x-forwarded-for': self.fake_ip}).json()
        uniqueid = user['uuid']
        print('User {}'.format(uniqueid))

        self.client.get('/api/catalogue/categories', headers={'x-forwarded-for': self.fake_ip})
        # all products in catalogue
        products = self.client.get('/api/catalogue/products', headers={'x-forwarded-for': self.fake_ip}).json()
        for i in range(2):
            item = None
            while True:
                item = choice(products)
                if item['instock'] != 0:
                    break

            # vote for item
            if randint(1, 10) <= 3:
                self.client.put('/api/ratings/api/rate/{}/{}'.format(item['sku'], randint(1, 5)),
                                headers={'x-forwarded-for': self.fake_ip})

            self.client.get('/api/catalogue/product/{}'.format(item['sku']), headers={'x-forwarded-for': self.fake_ip})
            self.client.get('/api/ratings/api/fetch/{}'.format(item['sku']), headers={'x-forwarded-for': self.fake_ip})
            self.client.get('/api/cart/add/{}/{}/1'.format(uniqueid, item['sku']),
                            headers={'x-forwarded-for': self.fake_ip})

        cart = self.client.get('/api/cart/cart/{}'.format(uniqueid), headers={'x-forwarded-for': self.fake_ip}).json()
        item = choice(cart['items'])
        self.client.get('/api/cart/update/{}/{}/2'.format(uniqueid, item['sku']),
                        headers={'x-forwarded-for': self.fake_ip})

        # country codes
        try:
            code = choice(self.client.get('/api/shipping/codes', headers={'x-forwarded-for': self.fake_ip}).json())
            city = choice(self.client.get('/api/shipping/cities/{}'.format(code['code']),
                                      headers={'x-forwarded-for': self.fake_ip}).json())
            print('code {} city {}'.format(code, city))
            shipping = self.client.get('/api/shipping/calc/{}'.format(city['uuid']),
                                       headers={'x-forwarded-for': self.fake_ip}).json()
            shipping['location'] = '{} {}'.format(code['name'], city['name'])
            print('Shipping {}'.format(shipping))
            # POST
            cart = self.client.post('/api/shipping/confirm/{}'.format(uniqueid), json=shipping,
                                    headers={'x-forwarded-for': self.fake_ip}).json()
            print('Final cart {}'.format(cart))
            order = self.client.post('/api/payment/pay/{}'.format(uniqueid), json=cart,
                                     headers={'x-forwarded-for': self.fake_ip}).json()
            print('Order {}'.format(order))
        except JSONDecodeError as err:
            logging.warning("JSON error.")

    @task
    def error(self):
        if os.environ.get('ERROR') == '1':
            print('Error request')
            cart = {'total': 0, 'tax': 0}
            self.client.post('/api/payment/pay/partner-57', json=cart, headers={'x-forwarded-for': self.fake_ip})
