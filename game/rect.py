"""
   Pygame uses Rect objects to store and manipulate rectangular areas. A Rect
   can be created from a combination of left, top, width, and height values.
   Rects can also be created from python objects that are already a Rect or
   have an attribute named "rect".

   Any Pygame function that requires a Rect argument also accepts any of these
   values to construct a Rect. This makes it easier to create Rects on the fly
   as arguments to functions.

   The Rect functions that change the position or size of a Rect return a new
   copy of the Rect with the affected changes. The original Rect is not
   modified. Some methods have an alternate "in-place" version that returns
   None but effects the original Rect. These "in-place" methods are denoted
   with the "ip" suffix.

   The Rect object has several virtual attributes which can be used to move and
   align the Rect:

   ::

       x,y
       top, left, bottom, right
       topleft, bottomleft, topright, bottomright
       midtop, midleft, midbottom, midright
       center, centerx, centery
       size, width, height
       w,h

   All of these attributes can be assigned to:

   ::

       rect1.right = 10
       rect2.center = (20,30)

   Assigning to size, width or height changes the dimensions of the rectangle;
   all other assignments move the rectangle without resizing it. Notice that
   some attributes are integers and others are pairs of integers.

   If a Rect has a nonzero width or height, it will return True for a nonzero
   test. Some methods return a Rect with 0 size to represent an invalid
   rectangle.

   The coordinates for Rect objects are all integers. The size values can be
   programmed to have negative values, but these are considered illegal Rects
   for most operations.

   There are several collision tests between other rectangles. Most python
   containers can be searched for collisions against a single Rect.

   The area covered by a Rect does not include the right- and bottom-most edge
   of pixels. If one Rect's bottom border is another Rect's top border (i.e.,
   rect1.bottom=rect2.top), the two meet exactly on the screen but do not
   overlap, and ``rect1.colliderect(rect2)`` returns false.

   Though Rect can be subclassed, methods that return new rectangles are not
   subclass aware. That is, move or copy return a new :mod:`pygame.Rect`
   instance, not an instance of the subclass. This may change. To make subclass
   awareness work though, subclasses may have to maintain the same constructor
   signature as Rect.
"""


class Rect():
    def __init__(self, *args):
        """
        Rect(left, top, width, height) -> Rect
        Rect((left, top), (width, height)) -> Rect
        Rect(object) -> Rect
        """
        if len(args) == 4:
            # left, top, width, height
            self.left, self.top, self.width, self.height = args
        elif len(args) == 2:
            # (left, top), (width, height)
            self.left, self.top = args[0]
            self.width, self.height = args[1]
        elif len(args) == 1:
            self.left = args[0].left
            self.top = args[0].top
            self.width = args[0].width
            self.height = args[0].height
        else:
            raise ValueError("Invalid arguments passed to Rect initializer")

    def __len__(self):
        return 4

    def __getitem__(self, index):
        if index == 0:
            return self.left
        elif index == 1:
            return self.top
        elif index == 2:
            return self.width
        elif index == 3:
            return self.height
        else:
            # TODO: fix this.
            # a rect object must support slicing, but python throws
            # an exception with using the the modulo operator on a slice.
            # this is just a hack...for now!
            # return self[index]      # max recursion error
            # return self[index % 4]  # type error
            return [self.left, self.top, self.width, self.height][index]

    def __iter__(self):
        yield self.left
        yield self.top
        yield self.width
        yield self.height

    def __repr__(self):
        return "pygame2.Rect(left=%.2f, top=%.2f, width=%.2f, height=%.2f)" % \
               (self.left, self.top, self.width, self.height)

    def __eq__(self, other):
        return list(self) == list(other)

    def __ne__(self, other):
        return not list(self) == list(other)

    @property
    def x(self):
        """
        Returns the left edge of the rect
        """
        return self.left

    @x.setter
    def x(self, value):
        self.left = value

    @property
    def y(self):
        """
        Returns the top edge of the rect
        """
        return self.top

    @y.setter
    def y(self, value):
        self.top = value

    @property
    def right(self):
        """
        Returns the x value which is the right edge of the rect
        """
        return self.left + self.width

    @right.setter
    def right(self, value):
        self.left = value - self.width

    @property
    def bottom(self):
        """
        Returns the y value of the bottom edge of the rect
        """
        return self.top + self.height

    @bottom.setter
    def bottom(self, value):
        self.top = value - self.height

    @property
    def topleft(self):
        """
        Returns a point (x, y) which is the top, left corner of the rect.
        """
        return (self.left, self.top)

    @topleft.setter
    def topleft(self, value):
        self.left, self.top = value

    @property
    def bottomleft(self):
        """
        Returns a point (x, y) which is the bottom, left corner of the rect.
        """
        return (self.left, self.bottom)

    @bottomleft.setter
    def bottomleft(self, value):
        self.left, self.bottom = value

    @property
    def topright(self):
        """
        Returns a point (x, y) which is the top, right corner of the rect.
        """
        return (self.right, self.top)

    @topright.setter
    def topright(self, value):
        self.right, self.top = value

    @property
    def bottomright(self):
        """
        Returns a point (x, y) which is the bottom, right corner of the rect.
        """
        return (self.right, self.bottom)

    @bottomright.setter
    def bottomright(self, value):
        self.right, self.bottom = value

    @property
    def centerx(self):
        """
        Returns the center x value of the rect
        """
        return self.left + self.width / 2

    @centerx.setter
    def centerx(self, value):
        self.left = value - self.width / 2

    @property
    def centery(self):
        """
        Returns the center y value of the rect
        """
        return self.top + self.height / 2

    @centery.setter
    def centery(self, value):
        self.top = value - self.height / 2

    @property
    def center(self):
        """
        Returns a point (x, y) which is the center of the rect.
        """
        return (self.centerx, self.centery)

    @center.setter
    def center(self, value):
        self.centerx, self.centery = value

    @property
    def midtop(self):
        """
        Returns a point (x, y) which is the midpoint of the top edge
        the rect.
        """
        return (self.centerx, self.top)

    @midtop.setter
    def midtop(self, value):
        self.centerx, self.top = value

    @property
    def midleft(self):
        """
        Returns a point (x, y) which is the midpoint of the left edge
        the rect.
        """
        return (self.left, self.centery)

    @midleft.setter
    def midleft(self, value):
        self.left, self.centery = value

    @property
    def midbottom(self):
        """
        Returns a point (x, y) which is the midpoint of the bottom edge
        the rect.
        """
        return (self.centerx, self.bottom)

    @midbottom.setter
    def midbottom(self, value):
        self.centerx, self.bottom = value

    @property
    def midright(self):
        """
        Returns a point (x, y) which is the midpoint of the right edge of
        the rect.
        """
        return (self.right, self.centery)

    @midright.setter
    def midright(self, value):
        self.right, self.centery = value

    @property
    def size(self):
        """
        Returns the width and height of the rect as a tuple (width, height)
        """
        return (self.width, self.height)

    @size.setter
    def size(self, value):
        self.width, self.height = value

    @property
    def w(self):
        """
        The width of the rect
        """
        return self.width

    @w.setter
    def w(self, value):
        self.width = value

    @property
    def h(self):
        """
        The height of the rect
        """
        return self.height

    @h.setter
    def h(self, value):
        self.height = value

    def clip(self, B):
        """
        Returns a new rectangle that is cropped to be completely inside
        the argument Rect. If the two rectangles do not overlap to
        begin with, a Rect with 0 size is returned.
        """
        # /* Left */
        if ((self.x >= B.x) and (self.x < (B.x + B.w))):
            x = self.x
        elif ((B.x >= self.x) and (B.x < (self.x + self.w))):
            x = B.x
        else:
            return Rect(self.x, self.y, 0, 0)

        # /* Right */
        if (((self.x + self.w) > B.x) and ((self.x + self.w) <= (B.x + B.w))):
            w = (self.x + self.w) - x
        elif (((B.x + B.w) > self.x) and ((B.x + B.w) <= (self.x + self.w))):
            w = (B.x + B.w) - x
        else:
            return Rect(self.x, self.y, 0, 0)

        # /* Top */
        if ((self.y >= B.y) and (self.y < (B.y + B.h))):
            y = self.y
        elif ((B.y >= self.y) and (B.y < (self.y + self.h))):
            y = B.y
        else:
            return Rect(self.x, self.y, 0, 0)

        # /* Bottom */
        if (((self.y + self.h) > B.y) and ((self.y + self.h) <= (B.y + B.h))):
            h = (self.y + self.h) - y
        elif (((B.y + B.h) > self.y) and ((B.y + B.h) <= (self.y + self.h))):
            h = (B.y + B.h) - y
        else:
            return Rect(self.x, self.y, 0, 0)

        return Rect(x, y, w, h)